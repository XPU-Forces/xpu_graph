import operator
from typing import List, Optional, Tuple

import torch
import torch.fx as fx
import torch_npu
from torch import fx, nn
from torch.fx.node import Node

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ...utils.check_ops import (
    check_add_op,
    check_getitem_op,
    check_norm_op,
    check_typecast_op,
)
from .check_npu_ops import check_npu_norm_op, check_npu_typecast_op
from .triton_kernel.fused_add_rmsnorm import fused_add_rmsnorm


class AddRmsnormOperation(nn.Module):
    def forward(self, input, residual, weight):
        return torch.ops.torch_npu_triton.fused_add_rmsnorm(input, residual, weight)


class FusedAddRmsnorm(Pattern):
    _opt_level = OptLevel.level2

    def _match_pattern(self, getitem_final_rmsnorm: Node) -> Optional[List[Node]]:
        if not check_getitem_op(getitem_final_rmsnorm):
            return None
        final_rmsnorm = getitem_final_rmsnorm.args[0]

        res_flag, op_name = check_norm_op(final_rmsnorm)
        if (not res_flag) and (not op_name == "rms_norm"):
            res_flag, op_name = check_npu_norm_op(final_rmsnorm)
            if (not res_flag) and (not op_name == "rms_norm"):
                return None

        to_node, weight_node = final_rmsnorm.args  # eps

        to_flag, to_ty = check_typecast_op(to_node)
        if not to_flag:
            to_flag, to_ty = check_npu_typecast_op(to_node)
            if not to_flag:
                return None
        new_residual = to_node.args[0]
        new_residual_dty = to_ty
        if new_residual_dty != torch.bfloat16:
            return None

        if not (check_add_op(new_residual)):
            return None

        add1, add2 = new_residual.args
        add1_flag, add1_ty = check_typecast_op(add1)
        if not add1_flag:
            add1_flag, add1_ty = check_npu_typecast_op(add1)
        add2_flag, add2_ty = check_typecast_op(add2)
        if not add2_flag:
            add2_flag, add2_ty = check_npu_typecast_op(add2)

        if not add1_flag or not add2_flag:
            return None

        # triton kernel only support torch.float32
        if add1_ty != torch.float32 or add2_ty != torch.float32:
            return None

        add1_inp = add1.args[0]
        add2_inp = add2.args[0]

        return [add1_inp, add2_inp, weight_node, final_rmsnorm, to_node, getitem_final_rmsnorm]

    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_add_rmsnorm", AddRmsnormOperation())

        for node in reversed(list(graph.nodes)):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue
            changed = True

            (add1_inp, residual, weight_node, final_rmsnorm, to_node, getitem_final_rmsnorm) = matched_nodes

            with graph.inserting_before(getitem_final_rmsnorm):
                fused_rms_node = graph.call_module(
                    "npu_triton_fused_add_rmsnorm",
                    args=(
                        add1_inp,
                        residual,
                        weight_node,
                    ),
                )

            with graph.inserting_after(getitem_final_rmsnorm):
                fused_rms_node_out = graph.call_function(operator.getitem, args=(fused_rms_node, 0))
                fused_to_node_out = graph.call_function(operator.getitem, args=(fused_rms_node, 1))

            getitem_final_rmsnorm.replace_all_uses_with(fused_rms_node_out)
            to_node.replace_all_uses_with(fused_to_node_out)

            nodes_to_remove = [
                getitem_final_rmsnorm,
                final_rmsnorm,
                to_node,
            ]
            for n in nodes_to_remove:
                graph.erase_node(n)

        return changed
