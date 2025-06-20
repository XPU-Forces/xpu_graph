import torch
from torch import nn, fx
from torch.fx.node import Node
import torch.fx as fx
from typing import Optional, Tuple, List
import torch_npu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.config import OptLevel
import operator
from ...utils.check_ops import (
    check_add_op,
    check_norm_op,
    check_typecast_op,
    check_getitem_op
)

"""
def forward(self, input, residual, arg486_1, arg487_1, weight=None):
    quant_matmul_v2_111 = input
    npu_dtype_cast_164 = residual
    arg485_1 = weight
    npu_dtype_cast_165 = torch.ops.npu.npu_dtype_cast.default(quant_matmul_v2_111, torch.float32);  quant_matmul_v2_111 = None
    npu_dtype_cast_166 = torch.ops.npu.npu_dtype_cast.default(npu_dtype_cast_164, torch.float32);  npu_dtype_cast_164 = None
    add_55 = torch.ops.aten.add.Tensor(npu_dtype_cast_165, npu_dtype_cast_166);  npu_dtype_cast_165 = npu_dtype_cast_166 = None
    npu_dtype_cast_167 = torch.ops.npu.npu_dtype_cast.default(add_55, torch.bfloat16);  add_55 = None
    npu_rms_norm_56 = torch.ops.npu.npu_rms_norm.default(npu_dtype_cast_167, arg485_1);  npu_dtype_cast_167 = arg485_1 = None
    getitem_476 = npu_rms_norm_56[0];  npu_rms_norm_56 = None
    
    index_select = torch.ops.aten.index_select.default(getitem_476, 0, arg486_1);  getitem_476 = arg486_1 = None
    t = torch.ops.aten.t.default(arg487_1);  arg487_1 = None
    mm = torch.ops.aten.mm.default(index_select, t);  index_select = t = None
    return mm
"""

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
            return None

        to_node, weight_node = final_rmsnorm.args  # eps
        to_flag, to_ty = check_typecast_op(to_node)
        if not to_flag:
            return None

        if to_ty == "to_copy":
            new_residual = to_node.args[0]
            new_residual_dty = to_node.kwargs['dtype']
            if (new_residual_dty != torch.bfloat16):
                return None
        elif to_ty == "to":
            new_residual, new_residual_dty = to_node.args
            if (new_residual_dty != torch.bfloat16):
                return None
        else:
            return None 
        
        if not (check_add_op(new_residual)):
            return None

        add1, add2 = new_residual.args
        if not (check_typecast_op(add1) and check_typecast_op(add2)):
            return None
        
        add1_inp = add1.args[0]
        add2_inp = add2.args[0]

        return [
            add1_inp, 
            add2_inp,
            weight_node,
            final_rmsnorm,
            to_node,
            getitem_final_rmsnorm
        ]


    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_add_rmsnorm", AddRmsnormOperation())

        for node in reversed(list(graph.nodes)):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue
            changed = True

            (add1_inp, residual, weight_node,
            final_rmsnorm, to_node, getitem_final_rmsnorm) = matched_nodes

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
                fused_rms_node_out = graph.call_function(
                        operator.getitem,
                        args=(fused_rms_node, 0)
                )
                fused_to_node_out = graph.call_function(
                        operator.getitem,
                        args=(fused_rms_node, 1)
                )
            
            getitem_final_rmsnorm.replace_all_uses_with(fused_rms_node_out)
            to_node.replace_all_uses_with(fused_to_node_out)
            

            nodes_to_remove = [
                getitem_final_rmsnorm,
                final_rmsnorm,
                to_node,
            ]
            for n in nodes_to_remove:
                graph.erase_node(n)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed