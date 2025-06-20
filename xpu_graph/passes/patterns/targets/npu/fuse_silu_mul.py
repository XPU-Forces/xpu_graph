import torch
from torch import nn, fx
from torch.fx.node import Node
import torch.fx as fx
from typing import Optional, Tuple, List
import torch_npu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    check_act_op,
    check_mul_op,
    check_slice_op
)

"""
def forward(input):
    d = input.shape[-1] // 2
    return torch.nn.functional.silu(input[..., :d]) * input[..., d:]

def call_triton_mul_silu(input):
    shape = input.shape
    triton_res = torch.empty_strided(([shape[0], shape[1]//2]), (shape[1]//2, 1), dtype=input.dtype, device=input.device)
    core = shape[0]
    triton_mul_silu[core, 1, 1](triton_res, input, BLOCK_SIZE=8192, N=shape[-1])
    return XTensor(triton_res)
"""

from .triton_kernel.fused_silu_mul import fused_silu_mul

class AddSilumulOperation(nn.Module):
    def forward(self, input):
        return torch.ops.torch_npu_triton.fused_silu_mul(input)


class FusedSilumul(Pattern):
    _opt_level = OptLevel.level2

    def _match_pattern(self, final_mul: Node) -> Optional[List[Node]]:
        if not check_mul_op(final_mul):
            return None

        mul_op1, mul_op2 = final_mul.args

        op_flag, op_name = check_act_op(mul_op1)
        if not op_flag and op_name != "silu":
            return None
        
        mul_op1_slice = mul_op1.args
        if not check_slice_op(mul_op1_slice[0]):
            return None

        if not check_slice_op(mul_op2):
            return None
        
        mul_op1_input = mul_op1_slice[0].args[0]
        mul_op2_input = mul_op2.args[0]
        
        if mul_op1_input != mul_op2_input:
            return None

        return [
            mul_op1_input,
            final_mul
        ]

    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_silu_mul", AddSilumulOperation())

        for node in reversed(list(graph.nodes)):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue
            changed = True

            (mul_input, final_mul) = matched_nodes

            with graph.inserting_before(final_mul):
                fused_silu_mul_node = graph.call_module(
                    "npu_triton_fused_silu_mul",
                    args=(
                        mul_input,
                    ),
                )
            final_mul.replace_all_uses_with(fused_silu_mul_node)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed
