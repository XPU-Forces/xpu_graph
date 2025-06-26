import torch
from torch import fx
from torch.fx.node import map_arg

from xpu_graph import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.cat_utils import match_sub_list
from ..utils.check_ops import (
    check_cat_op,
    check_meta_2d,
    check_slice_op,
    check_stack_op,
    check_sum_op,
)

MAX_INT64 = 9223372036854775807


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


def try_fuse_slice_cat(graph_module: fx.GraphModule, node):
    is_cat, cat_axis = check_cat_op(node)
    if not is_cat:
        return False
    if not check_meta_2d(node):
        return False
    if cat_axis < 0:
        cat_axis += len(node.meta["val"].shape)
    if cat_axis == 0:
        return False
    ori_cat_input = node.args[0]
    replaced_map = {}
    end = 0
    for idx in range(len(ori_cat_input)):
        # Find all slice + stack ops
        # original: [B, D]xN --stack--> [N, B, D]
        # after: [B, D]xN --slice_cat--> [B,N*D] --unflatten-->  [B, N, D]
        if idx < end:
            continue
        cand_start, cand_end = match_sub_list(
            ori_cat_input[idx:], lambda val: check_slice_op(val) and check_meta_2d(val)
        )
        if cand_start == -1:
            break
        start = idx + cand_start
        end = idx + cand_end + 1
        n_list = ori_cat_input[start:end]
        is_slice, src_node, slice_param = validate_slice_operation(n_list)
        if not is_slice:
            continue

        with graph_module.graph.inserting_before(node):
            slice_cat_node = graph_module.graph.call_module(
                "fuse_slice_cat",
                args=(src_node, slice_param),
            )

        replaced_map[(start, end)] = slice_cat_node

    if len(replaced_map) == 0:
        return False

    with graph_module.graph.inserting_before(node):
        new_cat_input = []
        previous = 0
        for start, end in replaced_map:
            new_cat_input.extend(ori_cat_input[previous:start])
            new_cat_input.append(replaced_map[(start, end)])
            previous = end
        new_cat_input.extend(ori_cat_input[previous:])
        if len(new_cat_input) > 1:
            new_cat_node = graph_module.graph.call_function(
                torch.ops.aten.cat,
                args=(new_cat_input, cat_axis),
            )
        else:
            new_cat_node = new_cat_input[0]

    node.replace_all_uses_with(new_cat_node)
    return True


def _is_user_transposable_sum(node):
    if len(node.users) != 1:
        return False
    sum_node = next(iter(node.users))
    if not check_sum_op(sum_node) or len(sum_node.args) < 2:
        return False
    dim = sum_node.args[1]
    if dim != [0]:
        return False
    return True


def try_fuse_slice_stack(graph_module, node):
    if not check_stack_op(node) or (len(node.args) > 1 and node.args[1] != 0):
        return False
    ori_stack_input = node.args[0]
    replaced_map = {}
    end = 0
    for idx in range(len(ori_stack_input)):
        # Find all slice + stack ops
        # original: [B, D]xN --stack--> [N, B, D]
        # after: [B, D]xN --slice_cat--> [B,N*D] --unflatten-->  [B, N, D]
        if idx < end:
            continue
        cand_start, cand_end = match_sub_list(
            ori_stack_input[idx:], lambda val: check_slice_op(val) and check_meta_2d(val)
        )
        if cand_start == -1:
            break
        start = idx + cand_start
        end = idx + cand_end + 1
        n_list = ori_stack_input[start:end]
        is_slice, src_node, slice_param = validate_slice_operation(n_list)
        if not is_slice:
            continue

        with graph_module.graph.inserting_before(node):
            slice_cat_node = graph_module.graph.call_module(
                "fuse_slice_cat",
                args=(src_node, slice_param),
            )
            view_node = graph_module.graph.call_function(
                torch.ops.aten.unflatten.int, args=(slice_cat_node, -1, [len(n_list), -1])
            )

        replaced_map[(start, end)] = view_node

    if len(replaced_map) == 0:
        return False

    if len(replaced_map) == 1 and (0, len(ori_stack_input)) in replaced_map and _is_user_transposable_sum(node):
        slice_cat_node = replaced_map[(0, len(ori_stack_input))]
        sum_node = next(iter(node.users))
        sum_node.args = (slice_cat_node, [1], *sum_node.args[2:])
    else:
        with graph_module.graph.inserting_before(node):
            new_cat_input = []
            previous = 0
            for start, end in replaced_map:
                new_cat_input.extend(
                    map_arg(
                        ori_stack_input[previous:start],
                        lambda n: graph_module.graph.call_function(torch.ops.aten.unsqueeze, args=(n, 0)),
                    )
                )
                trans_node = graph_module.graph.call_function(
                    torch.ops.aten.transpose.int, args=(replaced_map[(start, end)], 0, 1)
                )
                new_cat_input.append(trans_node)
                previous = end
            new_cat_input.extend(
                map_arg(
                    ori_stack_input[previous:],
                    lambda n: graph_module.graph.call_function(torch.ops.aten.unsqueeze, args=(n, 0)),
                )
            )
            if len(new_cat_input) > 1:
                new_cat_node = graph_module.graph.call_function(
                    torch.ops.aten.cat,
                    args=(new_cat_input, 0),
                )
            else:
                new_cat_node = new_cat_input[0]

        node.replace_all_uses_with(new_cat_node)

    graph_module.graph.erase_node(node)
    return True


class FusedSliceCat(Pattern):
    """
    slice + cat -> fuse_slice_cat
    """

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        changed = False
        if not hasattr(graph_module, "fuse_slice_cat"):
            graph_module.add_submodule(
                "fuse_slice_cat",
                self.target_mod(),
            )
        # the inputs of cat are mixed with slice and other ops.
        for node in reversed(graph_module.graph.nodes):
            changed |= try_fuse_slice_cat(graph_module, node)
            if self._opt_level >= OptLevel.level2:
                # If level2, change slice+stack to slice_cat+transpose
                changed |= try_fuse_slice_stack(graph_module, node)

        return changed
