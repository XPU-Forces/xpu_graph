import torch
from torch import nn, fx
from torch.fx.node import Node
import torch.fx as fx
from typing import Optional, Tuple, List
import torch_npu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    check_div_op,
    check_expand_op,
    check_mul_op,
    check_sum_op,
)

"""
inputs:
N0, N1, N2, N3, N4 = 10, 15, 255, 4, 128
DTYPE = torch.float32
arg48_1 = torch.randn((N0, N1, N3, N4), dtype=DTYPE, device=DEV)
unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg48_1, 2)
arg49_1 = torch.randn((N0, N2, N3, N4), dtype=DTYPE, device=DEV)
unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg49_1, 1)
clamp_min =  torch.randn((N0, N1, 1, N3, 1), dtype=DTYPE, device=DEV)
clamp_min_1 =  torch.randn((N0, 1, N2, N3, 1), dtype=DTYPE, device=DEV)

outputs:
mul_2 = torch.zeros((N0, N1, N2, N3), device=DEV, dtype=DTYPE)

def torch_func(unsqueeze_2, clamp_min, unsqueeze_3, clamp_min_1):
    expand = torch.ops.aten.expand.default(clamp_min, [N0, N1, 1, N3, N4])
    expand_1 = torch.ops.aten.expand.default(clamp_min_1, [N0, 1, N2, N3, N4])
    div = torch.ops.aten.div.Tensor(unsqueeze_2, expand)
    div_1 = torch.ops.aten.div.Tensor(unsqueeze_3, expand_1)
    mul_1 = torch.ops.aten.mul.Tensor(div, div_1)
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_1, [4])
    mul_2 = torch.ops.aten.mul.Tensor(sum_3, 0.5)
    return mul_2
"""

from .triton_kernel.fused_div_mul_sum import fused_div_mul_sum

class DivMulSumOperation(nn.Module):
    def forward(self, div1_input, div1_divisor, div2_input, div2_divisor):
        return torch.ops.torch_npu_triton.fused_div_mul_sum(div1_input, div1_divisor, div2_input, div2_divisor)

class FusedDivMulSum(Pattern):
    _opt_level = OptLevel.level2

    def _match_expand_div_pair(self, div_node: Node) -> Optional[Tuple[Node, Node]]:
        if not (check_div_op(div_node)):
            return None

        _, divisor = div_node.args
        if not (check_expand_op(divisor)):
            return None

        base_input = divisor.args[0]
        expand_shape = divisor.args[1]
        return (divisor, base_input, expand_shape)

    def _match_pattern(self, final_mul: Node) -> Optional[List[Node]]:
        # 匹配完整的操作链：div → div → mul → sum → mul_scalar from Deberta Model
        if not (check_mul_op(final_mul)):
            return None
        if not isinstance(final_mul.args[1], (float, int)):
            return None

        sum_node, scalar_val = final_mul.args
        if not (check_sum_op(sum_node)):
            return None
        if (sum_node.args[1] != [4]):
            return None

        mul_node = sum_node.args[0]
        if not (check_mul_op(mul_node)):
            return None

        div1, div2 = mul_node.args
        expand1_info = self._match_expand_div_pair(div1)
        if not expand1_info:
            return None
        expand1_node, base1_input, expand1_shape = expand1_info

        expand2_info = self._match_expand_div_pair(div2)
        if not expand2_info:
            return None
        expand2_node, base2_input, expand2_shape = expand2_info

        return [
            final_mul, sum_node, mul_node, 
            div1, expand1_node,
            div2, expand2_node
        ]


    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_div_mul_sum", DivMulSumOperation())

        for node in reversed(list(graph.nodes)):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue
            changed = True

            (final_mul, sum_node, mul_node, 
             div1, expand1_node, 
             div2, expand2_node) = matched_nodes

            div1_dividend = div1.args[0]
            div1_base_input = expand1_node.args[0]
            expand1_shape = expand1_node.args[1]

            div2_dividend = div2.args[0]
            div2_base_input = expand2_node.args[0]
            expand2_shape = expand2_node.args[1]

            with graph.inserting_before(final_mul):
                fused_node = graph.call_module(
                    "npu_triton_fused_div_mul_sum",
                    args=(
                        div1_dividend,   # unsqueeze_2 的输出
                        div1_base_input, # clamp_min 原始输入
                        # expand1_shape,   # [10,15,1,4,128]
                        div2_dividend,   # unsqueeze_3 的输出
                        div2_base_input, # clamp_min_1 原始输入
                        # expand2_shape,   # [10,1,255,4,128]
                        # sum_dims,        # [4]
                        # scalar_val       # 0.5
                    ),
                )

            final_mul.replace_all_uses_with(fused_node)

            nodes_to_remove = [
                final_mul, sum_node, mul_node,
                div1, expand1_node,
                div2, expand2_node
            ]
            for n in nodes_to_remove:
                if len(n.users) == 0:
                    graph.erase_node(n)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed
