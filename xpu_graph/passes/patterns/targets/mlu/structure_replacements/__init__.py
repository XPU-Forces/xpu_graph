import torch
import torch.fx as fx
import torch_mlu

from ..triton_kernel.fused_serial_mm_2dot import fuse_serial_mm_2dot
from ..triton_kernel.fused_serial_mm_3dot import fuse_serial_mm_3dot
from ..triton_kernel.fused_slice import fused_slice_low
from ..triton_kernel.fused_slice_cat import fused_slice_cat
from ..triton_kernel.fused_slice_v2 import fused_slice_low_v2
from ..triton_kernel.fused_sum_3d import fused_sum_3d_input
from ..triton_kernel.get_mlu_devinfo import get_device_properties
from .dense_layer_modules import BatchDenseLayerModule, DenseLayerModule
from .flash_attention_modules import FlashAttentionModule, can_fuse_fa
from .norm_modules import LayerNormModule, RMSNormModule


class FuseSliceModule(torch.nn.Module):
    def __init__(self, slices_index):
        super().__init__()
        device = torch.mlu.current_device()
        self.slices_index = torch.tensor(slices_index, dtype=torch.int32, device="mlu:" + str(device))

    def forward(self, input_tensor, slice_len):
        is_nd = False
        pre_dim = None
        if len(input_tensor.shape) > 2:
            is_nd = True
            pre_dim = list(input_tensor.shape)[:-1]
            sn = input_tensor.shape[-1]
            input_tensor = input_tensor.view(-1, sn)

        output = fused_slice_low(
            input_tensor,
            self.slices_index,
            slice_len,
            input_tensor.shape[0],
            input_tensor.stride(0),
        )
        if is_nd:
            new_shape = [len(self.slices_index)] + pre_dim + [slice_len]
            return output.view(new_shape)
        else:
            return output.view(len(self.slices_index), input_tensor.shape[0], slice_len)


class FuseSliceCatSameInputModule(torch.nn.Module):
    def forward(self, input_tensor, slices):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        indices = [i for start, end in slices for i in range(start, end)]
        rows, _ = input_tensor.shape
        indices_tensor = torch.tensor(indices, dtype=torch.int32, device=input_tensor.device)
        return fused_slice_cat(
            input_tensor,
            indices_tensor,
            rows,
            len(indices),
            input_tensor.stride(0),
        )


class FuseSliceCatSameInputModule_v2(torch.nn.Module):
    def __init__(self, many_slices):
        super().__init__()
        self.use_triton = False  # True
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        device = torch.mlu.current_device()

        if self.use_triton:
            with unset_fake_temporarily():
                indices = []
                self.total_output = []
                slices_index = [0]
                for slices in many_slices:
                    sum_ = 0
                    for slice_ in slices:
                        start, end = slice_
                        indices += range(start, end)
                        sum_ += end - start
                    slices_index.append(sum_ + slices_index[-1])
                    self.total_output.append(sum_)
                self.indices_tensor = torch.tensor(indices, dtype=torch.int32, device=device)
                self.indices_len = len(indices)

                self.slices_index = torch.tensor(slices_index[:-1], dtype=torch.int32, device=device)
                self.total_output_tensor = torch.tensor(self.total_output, dtype=torch.int32, device=device)
        else:
            with unset_fake_temporarily():
                num_output = 0
                output_ids = []
                input_dims = []
                output_dims = []
                output_offsets = []
                input_offsets = []
                for slices in many_slices:
                    sum_ = 0
                    for slice_ in slices:
                        start, end = slice_
                        slice_len = end - start
                        input_offsets.append(start)
                        input_dims.append(slice_len)
                        output_ids.append(num_output)
                        output_offsets.append(sum_)
                        sum_ += slice_len
                    output_dims.append(sum_)
                    num_output += 1
                self.input_dims = torch.tensor(input_dims, device=device, dtype=torch.int32)
                self.input_offsets = torch.tensor(input_offsets, device=device, dtype=torch.int32)
                self.output_dims = torch.tensor(output_dims, device=device, dtype=torch.int32)
                self.output_offsets = torch.tensor(output_offsets, device=device, dtype=torch.int32)
                self.output_ids = torch.tensor(output_ids, device=device, dtype=torch.int32)
                self.output_dims_list = output_dims

    def forward(self, input_tensor, many_slices):
        if self.use_triton:
            output_total = fused_slice_cat(
                input_tensor,
                self.indices_tensor,
                input_tensor.shape[0],
                self.indices_len,
                input_tensor.stride(0),
            )
            outputs = fused_slice_low_v2(
                output_total,
                self.slices_index,
                self.total_output_tensor,
                self.total_output,
            )
            return outputs
        else:
            return torch.ops.torch_mlu.emb_concat(
                input_tensor,
                self.input_offsets,
                self.input_dims,
                self.output_ids,
                self.output_offsets,
                self.output_dims,
                self.output_dims_list,  # for fake tensor
            )


class ComboSumModule(torch.nn.Module):
    def forward(self, input_list, dim):
        fused_inputs = []
        fused_indices = []
        outputs = [None] * len(input_list)
        for idx, input_tensor in enumerate(input_list):
            shape = input_tensor.shape
            if shape[1] > 64 or shape[2] > 64:
                # outputs[idx] = input_tensor.sum(dim=dim[0])
                outputs[idx] = fused_sum_3d_input([input_tensor], dim[0])[0]
            else:
                fused_inputs.append(input_tensor)
                fused_indices.append(idx)

        if fused_inputs != []:
            fused_results = fused_sum_3d_input(fused_inputs, dim[0])
            for idx, fused_result in zip(fused_indices, fused_results):
                outputs[idx] = fused_result
        return outputs


class FusedDenseTower2Replacement(torch.nn.Module):
    def forward(
        self,
        tinyffn_input,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        act_mode_up,
        act_mode_down,
        is_transb_up,
        is_transb_down,
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    ):
        if not tinyffn_input.is_contiguous():
            tinyffn_input = tinyffn_input.contiguous()
        if not up_fc_weight.is_contiguous():
            up_fc_weight = up_fc_weight.contiguous()
        if not down_proj_weight.is_contiguous():
            down_proj_weight = down_proj_weight.contiguous()
        output = fuse_serial_mm_2dot(
            tinyffn_input,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            is_up_bias,
            is_down_bias,
            is_up_act,
            is_down_act,
        )
        return output


def can_fuse_dense_tower2(
    tinyffn_input,
    up_fc_weight,
    up_fc_bias,
    down_proj_weight,
    down_proj_bias,
    act_mode_up,
    act_mode_down,
    is_transb_up,
    is_transb_down,
    is_up_bias,
    is_down_bias,
    is_up_act,
    is_down_act,
):
    K1, N1 = up_fc_weight.shape
    N2, O2 = down_proj_weight.shape

    if N1 != N2:
        return False, []

    size_of_dtype = 2
    input_dtype = tinyffn_input.dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    # Note: all weights should accommodate the wram size (512KB)
    return (K1 * N1 + N2 * O2 + N1 + O2) <= (512 / size_of_dtype * 1024)


class FusedDenseTower3Replacement(torch.nn.Module):
    def forward(
        self,
        first_input,
        first_weight,
        first_bias,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        act_mode_first,
        act_mode_up,
        act_mode_down,
        is_transb_first,
        is_transb_up,
        is_transb_down,
        is_first_bias,
        is_up_bias,
        is_down_bias,
        is_first_act,
        is_up_act,
        is_down_act,
    ):
        if not first_input.is_contiguous():
            first_input = first_input.contiguous()
        if not first_weight.is_contiguous():
            first_weight = first_weight.contiguous()
        if not up_fc_weight.is_contiguous():
            up_fc_weight = up_fc_weight.contiguous()
        if not down_proj_weight.is_contiguous():
            down_proj_weight = down_proj_weight.contiguous()
        output = fuse_serial_mm_3dot(
            first_input,
            first_weight,
            first_bias,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            is_first_bias,
            is_up_bias,
            is_down_bias,
            is_first_act,
            is_up_act,
            is_down_act,
        )
        return output


def can_fuse_dense_tower3(
    first_input,
    first_weight,
    first_bias,
    up_fc_weight,
    up_fc_bias,
    down_proj_weight,
    down_proj_bias,
    act_mode_first,
    act_mode_up,
    act_mode_down,
    is_transb_first,
    is_transb_up,
    is_transb_down,
    is_first_bias,
    is_up_bias,
    is_down_bias,
    is_first_act,
    is_up_act,
    is_down_act,
):
    K1, N1 = first_weight.shape
    N1, N2 = up_fc_weight.shape
    N2, N3 = down_proj_weight.shape

    size_of_dtype = 2
    input_dtype = first_input.dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    # Note: all weights should accommodate the wram size (512KB)
    return (K1 * N1 + N1 * N2 + N2 * N3 + N1 + N2 + N3) <= (512 / size_of_dtype * 1024)


def get_structure_replacements(config):
    return {
        "CustomRMSNorm": RMSNormModule,
        "CustomLayerNorm": LayerNormModule,
        "FusedSlice": FuseSliceModule,
        "FusedCatSlice": FuseSliceCatSameInputModule,
        "FusedSliceStackSum": FuseSliceCatSameInputModule,
        "FusedMultipleSliceCat": FuseSliceCatSameInputModule_v2,
        "ComboSum3dInp": ComboSumModule,
        "CustomDenseLayer": DenseLayerModule,
        "CustomBatchDenseLayer": BatchDenseLayerModule,
        "FusedDenseTower2": (FusedDenseTower2Replacement, can_fuse_dense_tower2),
        "FusedDenseTower3": (FusedDenseTower3Replacement, can_fuse_dense_tower3),
        "FlashAttention": (FlashAttentionModule, can_fuse_fa),
    }
