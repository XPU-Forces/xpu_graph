from typing import List, Tuple

import torch
import torch_mlu
import triton
import triton.language as tl


@triton.jit
def mlu_triton_slice_low_kernel_v2(
    input_ptr,
    output_ptrs,
    start_indices_ptr,
    slice_lens_ptr,
    input_row,
    input_stride,
    TOTAL_JOBS: tl.constexpr = 16,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 128,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)

    for slice_idx in tl.static_range(TOTAL_JOBS):
        if slice_idx % program_dim == program_id:
            output_ptr = output_ptrs[slice_idx]
            start_index = tl.load(start_indices_ptr + slice_idx)
            slice_len = tl.load(slice_lens_ptr + slice_idx)
            loop = (input_row + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
            loop_c = (slice_len + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
            for l in range(loop):
                for l_c in range(loop_c):
                    input_block_ptr = tl.make_block_ptr(
                        base=input_ptr,
                        shape=(input_row, slice_len + start_index),
                        strides=(input_stride, 1),
                        offsets=(l * BLOCK_SIZE_R, start_index + l_c * BLOCK_SIZE_C),
                        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                        order=(1, 0),
                    )
                    value = tl.load(input_block_ptr, boundary_check=(1, 0), padding_option=0)
                    output_block_ptr = tl.make_block_ptr(
                        base=output_ptr,
                        shape=(input_row, slice_len),
                        strides=(slice_len, 1),
                        offsets=(l * BLOCK_SIZE_R, l_c * BLOCK_SIZE_C),
                        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                        order=(1, 0),
                    )
                    tl.store(output_block_ptr, value, boundary_check=(1, 0))


@torch.library.custom_op("torch_mlu_triton::fused_slice_low_v2", mutates_args=())
def fused_slice_low_v2(
    src_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_lens: torch.Tensor,
    slice_lens_list: List[int],
) -> List[torch.Tensor]:
    n_rows = src_tensor.shape[0]
    input_stride = src_tensor.stride(0)
    block_size_r = n_rows
    block_size_c = 64
    size_of_dtype = 2
    if src_tensor.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 384 * 1024
    if block_size_r * block_size_c * size_of_dtype > nram_limit:
        block_size_r = nram_limit // size_of_dtype // block_size_c
    block_size_r = min(n_rows, block_size_r)

    processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
    grid = (processor_count, 1, 1)
    num_slices = len(start_indices)

    outputs = []
    for s in slice_lens_list:
        output = torch.empty(
            (src_tensor.shape[0], s),
            device=src_tensor.device,
            dtype=src_tensor.dtype,
        )
        outputs.append(output)

    mlu_triton_slice_low_kernel_v2[grid](
        src_tensor,
        tuple(outputs),
        start_indices,
        slice_lens,
        n_rows,
        input_stride,
        num_slices,
        block_size_r,
        block_size_c,
    )
    return outputs


@fused_slice_low_v2.register_fake
def fused_slice_low_v2_fake(
    src_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_lens: torch.Tensor,
    slice_lens_list: List[int],
) -> List[torch.Tensor]:
    outputs = []
    for s in slice_lens_list:
        output = torch.empty(
            (src_tensor.shape[0], s),
            device=src_tensor.device,
            dtype=src_tensor.dtype,
        )
        outputs.append(output)
    return outputs
