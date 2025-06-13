from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import (
    check_cat_op,
    check_slice_op,
    check_where_op,
    check_zeros_op,
    get_shape,
)
from ...utils.match_sub_list import match_sub_list
from .combo_cat_utils import find_longest_same_input, find_longest_same_shape_sequence

MINI_LEN = 3


def match_slice_where(val):
    if len(val.users) != 1:
        return False
    if not check_where_op(val):
        return False
    condition, x, y = val.args[:3]
    if not check_zeros_op(x):
        return False
    if not check_slice_op(y):
        return False
    return True


class ComboSliceWhereCat(Pattern):
    _opt_level = OptLevel.level1
    """
    slice_1 -> where_1 ----\
    slice_2 -> where_2 ------> cat -> output
    slice_3 -> where_3 ----/
    other   ---------------/
    ---->
    slice_1 ----\
    slice_2 -----> cat -> where -> cat -> output
    slice_3 ----/              /
    other   ------------------/
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and (node.target == torch.ops.aten.cat.default or node.target == torch.ops.aten.stack.default)
        ]
        for node in candidates:
            is_stack = False
            if node.target == torch.ops.aten.stack.default:
                is_stack = True
            ori_cat_input = node.args[0]
            if is_stack:
                axis = 0
            else:
                axis = node.args[1]
            if len(ori_cat_input) < MINI_LEN:
                continue
            start, end = match_sub_list(
                ori_cat_input,
                match_slice_where,
            )
            if end - start + 1 < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_input(ori_cat_input, start, end, MINI_LEN)
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            n_list = ori_cat_input[best_start : best_end + 1]
            where_inputs = [n.args[2] for n in n_list]
            condition_input = n_list[0].args[0]
            with graph_module.graph.inserting_before(node):
                slice_cat_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(where_inputs, axis),
                    name=node.name + "_combo_slice_where_cat_replacement1",
                )

                # where_op = torch.zeros_like(slice_cat_node)
                zeros_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.zeros_like.default,
                    args=(slice_cat_node,),
                    name=node.name + "_combo_slice_where_cat_zeros",
                    kwargs=dict(n_list[0].args[1].kwargs),
                )
                condition_expanded_to_zero = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.expand_as.default,
                    args=(condition_input, n_list[0].args[1]),
                    name=node.name + "_combo_slice_where_cat_condition_expanded_to_zero",
                )
                repeat_dim = [1] * len(get_shape(n_list[0]))
                repeat_dim[axis] = len(n_list)
                repeated_condition = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.repeat.default,
                    args=(condition_expanded_to_zero, repeat_dim),
                    name=node.name + "_combo_slice_where_cat_repeated_condition",
                )
                where_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.where.self,
                    args=(repeated_condition, zeros_node, slice_cat_node),
                    name=node.name + "_combo_slice_where_cat_replacement2",
                )

            new_cat_input = ori_cat_input[:best_start] + [where_node] + ori_cat_input[best_end + 1 :]
            last_node = where_node
            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_combo_slice_where_cat_replacement3",
                    )
                last_node = cat_node

            if is_stack:
                with graph_module.graph.inserting_before(node):
                    view_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.view.default,
                        args=(last_node, get_shape(node)),
                        kwargs={},
                    )
                last_node = view_node
            node.replace_all_uses_with(last_node)
            print(graph_module.graph)
            changed = True
        return changed
