import operator
from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

from ...utils.check_ops import check_cat_op, check_mul_op, check_slice_op, get_shape
from ...utils.match_sub_list import match_sub_list
from .combo_cat_utils import find_longest_same_input, find_longest_same_shape_sequence

MINI_LEN = 5


def match_mul(val):
    if len(val.users) == 1 or (len(val.users) == 2 and any(user.op == "output" for user in val.users)):
        if not check_mul_op(val):
            return False
        x, y = val.args[:2]
        if not check_slice_op(y):
            return False
        if len(get_shape(y)) != 2:
            return False
        if len(get_shape(x)) != 2:
            return False
        return True
    else:
        return False


class ComboMulCat(Pattern):
    _opt_level = OptLevel.level1
    _pattern_group = PatternGroup.GROUP1
    """
    %mul = mul.Tensor(cond, slice1)
    %mul_1 = mul.Tensor(cond, slice2)
    %mul_2 = mul.Tensor(cond, slice3)
    %mul_3 = mul.Tensor(cond, slice4)
    %cat = cat.default([%mul, %mul_1, %mul_2, %mul_3], -1)
    ---->
    %cat_data = cat.default([slice1, slice2, slice3, slice4], -1)
    %result = mul.Tensor(cat_data, cond)
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and (node.target == torch.ops.aten.cat.default)  # or node.target == torch.ops.aten.stack.default)
        ]
        for node in candidates:
            ori_cat_input = node.args[0]
            if len(ori_cat_input) < MINI_LEN:
                continue
            best_start, best_end = match_sub_list(
                ori_cat_input,
                match_mul,
            )
            if best_end - best_start + 1 < MINI_LEN:
                continue
            # check cond
            best_start, best_end = find_longest_same_input(ori_cat_input, best_start, best_end)
            if best_end - best_start + 1 < MINI_LEN:
                continue
            # check slice
            best_start, best_end = find_longest_same_shape_sequence(
                [n.args[1] for n in ori_cat_input], best_start, best_end
            )
            if best_end - best_start + 1 < MINI_LEN:
                continue
            axis = node.args[1] if len(node.args) > 1 else 0
            """
            if node.target == torch.ops.aten.stack.default:
                new_cat_node = convert_stack_to_cat(node)
            else:
                new_cat_node = node
            """

            n_list = ori_cat_input[best_start : best_end + 1]
            mul_inputs = [n.args[1] for n in n_list]
            condition_input = n_list[0].args[0]
            shape_input = get_shape(n_list[0].args[1])
            with graph_module.graph.inserting_before(node):
                input_node1 = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(mul_inputs, axis),
                    name=node.name + "_combo_mul_cat_replacement1",
                )
                if get_shape(condition_input)[axis] != 1:
                    if axis == 0:
                        new_shape = [len(n_list), shape_input[0], shape_input[1]]
                        new_shape1 = [-1, shape_input[1]]
                    else:
                        new_shape = [shape_input[0], len(n_list), shape_input[1]]
                        new_shape1 = [shape_input[0], -1]
                        axis = 1
                    input_node1_view = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.view.default,
                        args=(input_node1, new_shape),
                        name=node.name + "_combo_mul_cat_replacement2",
                    )
                    input_node2 = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.unsqueeze.default,
                        args=(condition_input, axis),
                        name=node.name + "_combo_mul_cat_replacement3",
                    )
                    mul_node_beforeview = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.mul.Tensor,
                        args=(input_node1_view, input_node2),
                        name=node.name + "_combo_mul_cat_replacement4",
                    )
                    mul_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.view.default,
                        args=(mul_node_beforeview, new_shape1),
                        name=node.name + "_combo_mul_cat_replacement5",
                    )
                else:
                    mul_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.mul.Tensor,
                        args=(input_node1, condition_input),
                        name=node.name + "_combo_mul_cat_replacement6",
                    )
                # for output
                split_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.split.Tensor,
                    args=(mul_node, shape_input[axis], axis),
                    name=node.name + "_combo_mul_cat_replacement7",
                )
                for i in range(len(n_list)):
                    with graph_module.graph.inserting_after(split_node):
                        new_n = graph_module.graph.call_function(
                            operator.getitem,
                            args=(split_node, i),
                            kwargs={},
                        )
                        n_list[i].replace_all_uses_with(new_n)

            new_cat_input = ori_cat_input[:best_start] + [mul_node] + ori_cat_input[best_end + 1 :]
            last_node = mul_node
            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_combo_mul_cat_replacement8",
                    )
                last_node = cat_node
            node.replace_all_uses_with(last_node)
            changed = True
        return changed
