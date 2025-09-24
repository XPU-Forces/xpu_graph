import operator
from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

from ...utils.check_ops import check_mul_op, check_zeros_op, get_shape


class MulReplacementRB(nn.Module):
    def forward(self, a, inputs, axis):
        catted_inputs = torch.stack(inputs)
        catted_result = a.unsqueeze(0) * catted_inputs
        return catted_result.unbind()


class MulReplacementLB(nn.Module):
    def forward(self, a, inputs, axis):
        if a.shape[axis] == 1:
            catted_inputs = torch.cat(inputs, dim=axis)
            catted_result = a * catted_inputs
            split_sizes = [inp.shape[axis] for inp in inputs]
            results = torch.split(catted_result, split_sizes, dim=axis)
            return list(results)
        else:  # the same shape
            catted_inputs = torch.stack(inputs)
            catted_result = a.unsqueeze(0) * catted_inputs
            return catted_result.unbind()


class ComboMulP(Pattern):
    """
    mul(a, args1)
    mul(a, args2)
    mul(a, args3)
    mul(a, args4)
    ->
    mulreplacement(a, [args1, args2, args3, args4])
    """

    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        candidates = [node for node in graph_module.graph.nodes if node.op == "call_function" and check_mul_op(node)]
        graph_module.add_submodule("mlu_fused_mul_lb", MulReplacementLB())
        graph_module.add_submodule("mlu_fused_mul_rb", MulReplacementRB())
        mul_cond_lb: Dict[fx.Node, List[fx.Node]] = {}  # left broadcast
        mul_cond_rb: Dict[fx.Node, List[fx.Node]] = {}
        for node in candidates:
            x, y = node.args[:2]
            if isinstance(x, (int, float)):
                continue
            if isinstance(y, (int, float)):
                continue
            if len(get_shape(x)) > 2:
                continue
            shape_x = get_shape(x)
            shape_y = get_shape(y)
            if len(shape_x) != len(shape_y):
                continue

            placeholder_op = x
            input_op = y
            if x.op == "placeholder":
                placeholder_op = x
                input_op = y
            elif y.op == "placeholder":
                placeholder_op = y
                input_op = x
            else:
                continue

            placeholder_op_shape = get_shape(placeholder_op)
            input_op_shape = get_shape(input_op)
            if placeholder_op_shape[-1] == input_op_shape[-1]:
                axis = 0
            else:
                axis = -1
            key = (input_op, axis)

            if placeholder_op_shape[axis] == 1:
                if key not in mul_cond_rb:
                    mul_cond_rb[key] = []
                mul_cond_rb[key].append((node, placeholder_op))
            else:
                if key not in mul_cond_lb:
                    mul_cond_lb[key] = []
                mul_cond_lb[key].append((node, placeholder_op))

        for cond, nodes in mul_cond_lb.items():
            input_op, axis = cond
            if len(nodes) < 5:
                continue
            with graph_module.graph.inserting_after(input_op):
                mul_node = graph_module.graph.call_module(
                    "mlu_fused_mul_lb",
                    args=(input_op, [n[1] for n in nodes], axis),
                )
            with graph_module.graph.inserting_after(mul_node):
                for i in range(len(nodes)):
                    new_n = graph_module.graph.call_function(
                        operator.getitem,
                        args=(mul_node, i),
                        kwargs={},
                    )
                    nodes[i][0].replace_all_uses_with(new_n)
            is_modified = True

        for cond, nodes in mul_cond_rb.items():
            input_op, axis = cond
            if len(nodes) < 5:
                continue
            with graph_module.graph.inserting_after(input_op):
                mul_node = graph_module.graph.call_module(
                    "mlu_fused_mul_rb",
                    args=(input_op, [n[1] for n in nodes], axis),
                )
            with graph_module.graph.inserting_after(mul_node):
                for i in range(len(nodes)):
                    new_n = graph_module.graph.call_function(
                        operator.getitem,
                        args=(mul_node, i),
                        kwargs={},
                    )
                    nodes[i][0].replace_all_uses_with(new_n)
            is_modified = True
        return is_modified
