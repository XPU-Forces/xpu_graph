import torch
from torch import fx, nn

from xpu_graph import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import (
    check_cat_op,
    check_meta_2d,
    check_slice_op,
    check_stack_op,
    check_sum_op,
)
from ..utils.match_sub_list import match_sub_list

MAX_INT64 = 9223372036854775807


class MergeCatReplacement(nn.Module):
    def forward(self, input_tensor_list, cat_axis=0):
        return torch.cat(
            [
                (input_tensor if len(input_tensor.shape) == 3 else input_tensor.unsqueeze(0))
                for input_tensor in input_tensor_list
            ],
            axis=0,
        )


def validate_slice_operation(n_list):
    if len(n_list) < 2:
        return False, None, None
    slice_input = []
    slice_param = []
    slice_axis = []
    for n in n_list:
        slice_input.append(n.args[0])
        slice_axis.append(n.args[1])
        right = n.args[3]
        if right == MAX_INT64:
            right = slice_input[0].meta["val"].shape[-1]
        elif right < 0:
            right = slice_input[0].meta["val"].shape[-1] - (-right)
        slice_param.append((n.args[2], right))
    if slice_input.count(slice_input[0]) != len(slice_input):
        return False, None, None
    if slice_axis.count(1) != len(slice_axis):
        return False, None, None
    return True, slice_input[0], slice_param


def fuse_mixed_ops_and_cat(graph_module: fx.GraphModule):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_cat, cat_axis = check_cat_op(node)
        if not is_cat:
            continue
        if not check_meta_2d(node):
            continue
        if cat_axis == 0:
            continue
        ori_cat_input = node.args[0]
        start, end = match_sub_list(ori_cat_input, lambda val: check_slice_op(val) and check_meta_2d(val))
        n_list = node.args[0][start : end + 1]
        is_slice, src_node, slice_param = validate_slice_operation(n_list)
        if not is_slice:
            continue

        new_cat_input = ori_cat_input[:start]

        with graph_module.graph.inserting_before(node):
            slice_node = graph_module.graph.call_module(
                "fuse_slice_cat",
                args=(src_node, slice_param),
            )

        new_cat_input.append(slice_node)

        new_cat_input += ori_cat_input[end + 1 :]
        with graph_module.graph.inserting_before(node):
            cat_node = graph_module.graph.create_node(
                op="call_function",
                target=torch.ops.aten.cat.default,
                args=(new_cat_input, -1),
                name=node.name + "_replacement",
            )
        node.replace_all_uses_with(cat_node)
        slice_nodes = node.args[0]
        for slice_node in slice_nodes:
            if len(slice_node.users) == 0:
                graph_module.graph.erase_node(slice_node)
        graph_module.graph.erase_node(node)
        changed = True

    return changed


class FusedCatSlice(Pattern):
    """
    slice + cat -> fuse_slice_cat
    """

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        changed = False
        graph_module.add_submodule(
            "fuse_slice_cat",
            self.target_mod(),
        )
        # the inputs of cat are mixed with slice and other ops.
        changed = changed | fuse_mixed_ops_and_cat(graph_module)

        return changed
