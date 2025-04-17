import torch
from torch import nn, fx
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from typing import Callable
from xpu_graph import OptLevel
from ..utils.check_ops import (
    check_cat_op,
    check_slice_op,
    check_stack_op,
    check_meta_2d,
    check_sum_op,
    get_actual_node,
    check_getitem_op,
)
from ..utils.match_sub_list import match_sub_list
import operator

MAX_INT64 = 9223372036854775807


class MergeCatReplacement(nn.Module):
    def forward(self, input_tensor_list, cat_axis=0):
        return torch.cat(
            [
                (
                    input_tensor
                    if len(input_tensor.shape) == 3
                    else input_tensor.unsqueeze(0)
                )
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


def check_sum_enable(gm, node):
    if len(node.users) != 1:
        return False
    sum_node = next(iter(node.users))
    if not check_sum_op(sum_node):
        return False
    dim = sum_node.args[1]
    if dim != [0]:
        return False
    with gm.graph.inserting_before(sum_node):
        new_sum_node = gm.graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(sum_node.args[0], [1])
        )
        sum_node.replace_all_uses_with(new_sum_node)
        gm.graph.erase_node(sum_node)
    return True


def fuse_mixed_ops_and_stack(graph_module: fx.GraphModule):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        if not check_stack_op(node):
            continue
        ori_cat_input = node.args[0]
        start, end = match_sub_list(
            ori_cat_input, lambda val: check_slice_op(val) and check_meta_2d(val)
        )
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

        with graph_module.graph.inserting_before(node):
            batch_node = graph_module.graph.call_method("size", args=(slice_node, 0))
            reshape_node = graph_module.graph.call_function(
                torch.ops.aten.view.default,
                args=(slice_node, (batch_node, len(slice_param), -1)),
            )
            # skip trans is the next node is sum, and change sum dim
            if (
                (start == 0)
                and (end + 1 == len(ori_cat_input))
                and check_sum_enable(graph_module, node)
            ):
                new_cat_input.append(reshape_node)
            else:
                transpose_node = graph_module.graph.call_function(
                    torch.ops.aten.transpose.int, args=(reshape_node, 0, 1)
                )
                new_cat_input.append(transpose_node)

        new_cat_input += ori_cat_input[end + 1 :]

        if len(new_cat_input) == 1:
            with graph_module.graph.inserting_before(node):
                cat_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(new_cat_input, 0),
                    name=node.name + "_replacement",
                )
        else:
            with graph_module.graph.inserting_before(node):
                cat_node = graph_module.graph.call_module(
                    "reshape_cat_module", args=(new_cat_input, -1)
                )
        node.replace_all_uses_with(cat_node)
        slice_nodes = node.args[0]
        for slice_node in slice_nodes:
            if len(slice_node.users) == 0:
                graph_module.graph.erase_node(slice_node)
        graph_module.graph.erase_node(node)
        changed = True

    return changed


class FusedSliceStackSum(Pattern):
    _opt_level = OptLevel.level2
    """
    slice + stack -> fuse_slice_cat + reshape + trans
    trans + sum -> sum
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
        graph_module.add_submodule("reshape_cat_module", MergeCatReplacement())

        # the inputs of stack are mixed with slice and other ops.
        changed = changed | fuse_mixed_ops_and_stack(graph_module)

        return changed


def find_slice_cat(node):
    for name in ["fuse_slice_cat", "fuse_slice_cat_v2"]:
        if name in node.name:
            return True
    return False


def find_sum2d_input(candidates):
    sum_input_dict = {}
    for sum_node in candidates:
        if not check_meta_2d(sum_node):
            continue
        # don't keep dim
        if len(sum_node.args) > 2:
            continue
        input_node = get_actual_node(sum_node, 0)
        if not check_getitem_op(input_node):
            continue
        src_node = input_node.args[0]
        if not find_slice_cat(src_node):
            continue
        dim = sum_node.args[1]
        # only support dim=1/2
        if dim not in [[1], [2]]:
            continue
        src_node = input_node.args[0]
        key = src_node.name + str(dim)
        if key not in sum_input_dict:
            sum_input_dict[key] = []
        sum_input_dict[key].append(sum_node)
    return sum_input_dict


def partly_topo_sort(gm: fx.Graph, node: fx.Node):
    import queue

    que = queue.Queue()
    que.put(node)
    while not que.empty():
        cur = que.get()
        for user in cur.users:
            if user < cur:
                cur.append(user)
                que.put(user)


class ComboSum2d(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1
    """
    #input: 3d output:2d
    sum1(a), sum2(b)-> combo_sum(a,b)
    """

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        changed = False

        graph_module.add_submodule(
            "combo_sum",
            self.target_mod(),
        )
        candidates = [
            node
            for node in graph_module.graph.nodes
            if (node.op == "call_function" or node.op == "call_module")
            and node.target == torch.ops.aten.sum.dim_IntList
        ]
        sum_input_dict = find_sum2d_input(candidates)
        for key, sum_nodes in sum_input_dict.items():
            sum_node = sum_nodes[-1]
            with graph_module.graph.inserting_after(sum_node):
                new_nodes = graph_module.graph.call_module(
                    "combo_sum",
                    args=([s.args[0] for s in sum_nodes], sum_node.args[1]),
                )

            for idx, ori_node in enumerate(sum_nodes):
                with graph_module.graph.inserting_after(new_nodes):
                    idx_node = graph_module.graph.call_function(
                        operator.getitem, args=(new_nodes, idx)
                    )
                ori_node.replace_all_uses_with(idx_node)
                partly_topo_sort(graph_module, idx_node)
                graph_module.graph.erase_node(ori_node)
            changed = True

        return changed
