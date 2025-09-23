import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ...utils.check_ops import (
    check_cat_op,
    check_meta_2d,
    check_slice_op,
    check_stack_op,
    check_sum_op,
)
from ...utils.submodule_manager import register_new_submodule
from .triton_kernel.fused_slice_sum_cat import fuse_slice_sum_cat
from .triton_kernel.fused_sum_cat import fuse_sum_cat_2d, fuse_sum_cat_3d


class ConcatenateSumOperation2(nn.Module):
    def forward(self, inputs, sum_dim, cat_axis):
        if sum_dim != [1]:
            raise NotImplementedError("sum_dim must be 1")
        if cat_axis != -1:
            raise NotImplementedError("cat_axis must be -1")

        batch_size, _, feature_dim = inputs[0].shape
        device = inputs[0].device
        sequence_lengths = [tensor.shape[1] for tensor in inputs]
        max_sequence_length = max(sequence_lengths)

        lengths_tensor = torch.tensor(sequence_lengths, device=device, dtype=torch.int32)

        return mlu_triton_fuse_sum_cat_3d(
            inputs,
            batch_size,
            lengths_tensor,
            feature_dim,
            len(inputs),
            max_sequence_length,
        )


class ConcatenateSumOperation1(nn.Module):
    def forward(self, inputs, sum_dim, concat_mode, keep_dims, cat_axis, is_cat=True):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        sequence_lengths = [tensor.shape[1] for tensor in inputs]
        max_sequence_length = max(sequence_lengths)
        length_tensor = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)
        if keep_dims is True and cat_axis != 0:
            output = mlu_triton_fuse_sum_cat_3d(
                inputs,
                batch_size,
                length_tensor,
                1,
                len(inputs),
                max_sequence_length,
            )
        else:
            output = mlu_triton_fuse_sum_cat_2d(
                inputs,
                length_tensor,
                batch_size,
                max_sequence_length,
            )
            if is_cat and cat_axis == -1:
                output = output.reshape([-1])
        return output


class SliceSumCatOperation(nn.Module):
    def __init__(self, slice_param):
        """
        Args:
            slice_param (list of tuples): A list of slice indices, where each tuple
                                          contains (start_idx, end_idx) for slicing.
        """
        super().__init__()
        device = torch.mlu.current_device()

        slice_ = []
        for param in slice_param:
            slice_ += [param[0], param[1]]
        self.slice_tensor = torch.tensor(slice_, dtype=torch.int32, device="mlu:" + str(device))

        self.output_num = len(slice_param)
        self.start = min([s[0] for s in slice_param])
        self.end = max([s[1] for s in slice_param])
        self.slice_param_list = slice_param

    def forward(self, input):
        """
        Forward pass for the SliceSumCatOperation.

        Args:
            input (torch.Tensor): The input tensor of shape (batch, row, col).

        Returns:
            torch.Tensor: The output tensor of shape (batch, len(slice_param) * col). The processed tensor after slice -> sum -> cat operations.
        """
        batch, row, col = input.shape
        # Ensure the slicing range does not exceed 1024 for computational efficiency
        if (self.end - self.start) > 1024:
            target_tensors = []
            for slice_arg in self.slice_param_list:
                slice_tensor = input[:, slice_arg[0] : slice_arg[1], :]
                sum_tensor = torch.sum(slice_tensor, dim=[1])
                target_tensors.append(sum_tensor)
            return torch.cat(target_tensors, axis=-1)
        else:
            return fuse_slice_sum_cat(input, self.slice_tensor, self.output_num, self.end)


# This function identifies slice->sum patterns in a computation graph (gm).
# The detected pattern consists of:
#  - A slice operation (input: 3D, output: 3D)
#  - A sum operation (input: 3D, output: 2D)
# Both slice and sum nodes should have only one output to maintain a strict computational structure.
def find_slice_sum_pattern(gm):
    # Dictionary to store mapping from source nodes to sum nodes
    slice_dict = {}
    for node in gm.graph.nodes:
        ### Identify sum operation: input 3D, output 2D ###
        if not check_sum_op(node):
            continue
        if not check_meta_2d(node):
            continue
        # check sum dim
        if node.args[1] != [1]:
            continue
        # don't keep dim
        if len(node.args) > 2:
            continue
        if len(node.users) != 1:
            continue

        ### Identify slice operation: input 3D, output 3D ###
        slice_node = node.args[0]
        if not check_slice_op(slice_node):
            continue
        # check slice dim
        if slice_node.args[1] != 1:
            continue
        # This restriction is currently in place but will be removed in the future.
        if slice_node.args[2] != 0:
            continue
        if len(slice_node.users) != 1:
            continue

        src_node = slice_node.args[0]
        if src_node not in slice_dict:
            slice_dict[src_node] = []
        slice_dict[src_node].append(node)
    return slice_dict


def match_slice_dict(slice_dict, node):
    for src_node, s_list in slice_dict.items():
        if node in s_list:
            return True, src_node, s_list
    return False, None, None


def match_slice_sum_cat_pattern(cat_input, slice_dict):
    for idx in range(len(cat_input)):
        s = cat_input[idx]
        if not check_sum_op(s):
            continue
        is_match, src_node, s_list = match_slice_dict(slice_dict, s)
        if not is_match:
            continue
        # Identify a contiguous sequence of sum nodes
        sum_start = idx
        sum_end = idx
        # Search for sum nodes that are derived from the same slice operation
        for next_idx in range(idx + 1, len(cat_input)):
            next_s = cat_input[next_idx]
            if next_s in s_list:
                sum_end = next_idx
            else:
                break
        # Ensure there are at least two consecutive sum nodes
        if (sum_end - sum_start) < 2:
            continue
        match_sum_list = cat_input[sum_start : sum_end + 1]
        # Record the slice parameters from the matched sum nodes
        args = [(s.args[0].args[2], s.args[0].args[3]) for s in match_sum_list]
        return True, (sum_start, sum_end), src_node, args
    return False, None, None, None


def find_slice_sum_cat(gm: fx.GraphModule):
    changed = False
    slice_dict = find_slice_sum_pattern(gm)
    if slice_dict == {}:
        return changed
    # slice->sum->cat
    for node in gm.graph.nodes:
        is_cat, _ = check_cat_op(node)
        if not is_cat:
            continue
        ori_cat_input = node.args[0]
        is_match, range_, src_node, slice_params = match_slice_sum_cat_pattern(ori_cat_input, slice_dict)
        if not is_match:
            continue
        has_symint = False
        for slice_param in slice_params:
            has_symint = has_symint or not isinstance(slice_param[0], int) or not isinstance(slice_param[1], int)
        if has_symint:
            continue
        with gm.graph.inserting_before(node):
            module_name = register_new_submodule(
                gm,
                "mlu_triton_slice_sum_cat",
                SliceSumCatOperation,
                args=(slice_params,),
            )
            new_node = gm.graph.call_module(
                module_name,
                args=(src_node,),
            )
            new_cat_input = ori_cat_input[: range_[0]] + [new_node] + ori_cat_input[range_[1] + 1 :]
            concat_node = gm.graph.create_node(
                op="call_function",
                target=torch.ops.aten.cat.default,
                args=(new_cat_input, -1),
                name=node.name + "_1",
            )

        node.replace_all_uses_with(concat_node)
        gm.graph.erase_node(node)
        match_sum_list = ori_cat_input[range_[0] : range_[1] + 1]
        for s in match_sum_list:
            if len(s.users) == 0:
                gm.graph.erase_node(s)
        changed = True
    return changed


class FusedCatSum(Pattern):
    _opt_level = OptLevel.level2

    def process(self, gm: fx.GraphModule):
        changed = False
        changed1 = True
        while changed1:
            changed1 = find_slice_sum_cat(gm)
            changed = changed | changed1

        return changed
