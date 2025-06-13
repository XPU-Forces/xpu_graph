from typing import Optional

import torch
import torch_mlu
from torch import fx, nn
from torch.fx.subgraph_rewriter import replace_pattern

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_cat_op, check_meta_2d, check_mul_op, check_sum_op
from .triton_kernel.fused_mul_sum_cat import fused_mul_sum_cat_2inp


class FusedMulSumCatReplacement(nn.Module):
    def forward(
        self,
        mul_list,
    ):
        s1, s2 = mul_list[0].shape[1:]
        if any(x.shape[1:] != (s1, s2) for x in mul_list[1:]):
            return torch.cat(
                [torch.sum(mul_list[i] * mul_list[i + 1]) for i in range(0, len(mul_list), 2)],
                dim=1,
            )

        if len(mul_list) == 4:
            mul0, mul1, mul2, mul3 = mul_list
            mul0 = mul0.contiguous()
            mul1 = mul1.contiguous()
            mul2 = mul2.contiguous()
            mul3 = mul3.contiguous()
            return fused_mul_sum_cat_2inp(
                mul0,
                mul1,
                mul2,
                mul3,
            )

        batch_size = max(mul_list[0].shape[0], mul_list[1].shape[0])
        new_mul_list = [
            (mul_list[i].expand(batch_size, s1, s2) if mul_list[i].shape[0] == 1 else mul_list[i])
            for i in list(range(0, len(mul_list), 2)) + list(range(1, len(mul_list), 2))
        ]
        tmp = torch.cat(new_mul_list, dim=0).view(2, len(new_mul_list) // 2, batch_size, s1, s2)
        tmp = tmp[0] * tmp[1]  # [-1, batch, s1, s2]
        output = torch.sum(tmp, dim=2).permute(1, 0, 2).reshape(batch_size, -1)
        return output


def find_mul_sum_pattern(gm):
    # Dictionary to store mapping from source nodes to sum nodes
    sum_dict = {}
    candidates = [
        node for node in gm.graph.nodes if node.op == "call_function" and node.target == torch.ops.aten.sum.dim_IntList
    ]
    for node in candidates:
        ### Identify sum operation: input 3D, output 2D ###
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

        mul_node = node.args[0]
        if not check_mul_op(mul_node):
            continue
        if len(mul_node.users) != 1:
            continue

        if node not in sum_dict:
            sum_dict[node] = [mul_node.args[0], mul_node.args[1]]
    return sum_dict


def check_enable(sum_inputs, sum_dict):
    for sum_node in sum_inputs:
        if sum_node not in sum_dict:
            return False
    return True


def get_mul_inputs(sum_inputs):
    mul_inputs = []
    for sum_node in sum_inputs:
        mul_node = sum_node.args[0]
        mul_inputs += [mul_node.args[0], mul_node.args[1]]
    return mul_inputs


class FusedMulSumCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        graph_module.add_submodule("mlu_triton_mul_sum_cat_replacement", FusedMulSumCatReplacement())

        for node in graph_module.graph.nodes:
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat:
                continue
            sum_dict = find_mul_sum_pattern(graph_module)
            if len(node.args[0]) < 2:
                continue
            if not check_enable(node.args[0], sum_dict):
                continue
            mul_inputs = get_mul_inputs(node.args[0])
            with graph_module.graph.inserting_before(node):
                new_node = graph_module.graph.call_module(
                    "mlu_triton_mul_sum_cat_replacement",
                    args=(list(mul_inputs),),
                )
            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            changed = True

        return changed
