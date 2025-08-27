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
    check_mul_op,
    get_shape,
)
from ...utils.match_sub_list import match_sub_list
from .combo_cat_utils import find_longest_same_input, find_longest_same_shape_sequence

MINI_LEN = 3


def match_mul(val):
    if len(val.users) != 1:
        return False
    if not check_mul_op(val):
        return False
    #if get_shape(val.args[0]) != get_shape(val.args[1]):
    #    return False
    return True


class ComboWhereCat(Pattern):
    _opt_level = OptLevel.level1
    """
    %mul = mul.Tensor(%select_1, %logical_not)
    %mul_1 = mul.Tensor(%select_2, %logical_not) 
    %mul_2 = mul.Tensor(%select_3, %logical_not)
    %mul_3 = mul.Tensor(%select_4, %logical_not)
    %cat = cat.default([%mul, %mul_1, %mul_2, %mul_3], -1)
    ---->
    %cat_data = cat.default([%select_1, %select_2, %select_3, %select_4], -1)
    %result = mul.Tensor(%cat_data, %expanded_logical_not)
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        return False

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
            best_start, best_end = match_sub_list(
                ori_cat_input,
                match_mul,
            )
            if best_end - best_start + 1 < MINI_LEN:
                continue
            n_list = ori_cat_input[best_start : best_end + 1]
            mul_inputs1 = [n.args[0] for n in n_list]
            mul_inputs2 = [n.args[1] for n in n_list]
            condition_input = n_list[0].args[0]
            with graph_module.graph.inserting_before(node):
                input_node1 = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(mul_inputs1, axis),
                    name=node.name + "_combo_mul_cat_replacement1",
                )
                input_node2 = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(mul_inputs2, axis),
                    name=node.name + "_combo_mul_cat_replacement2",
                )
                mul_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.mul.Tensor,
                    args=(input_node1, input_node2),
                    name=node.name + "_combo_mul_cat_replacement3",
                )

            new_cat_input = ori_cat_input[:best_start] + [mul_node] + ori_cat_input[best_end + 1 :]
            last_node = mul_node 
            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_combo_mul_cat_replacement4",
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
            changed = True
        return changed
