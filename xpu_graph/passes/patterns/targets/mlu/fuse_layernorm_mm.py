from typing import Optional

import operator
import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_copy,
)

class FusedLayernormMMReplacement(nn.Module):
    def forward(
        self, inputs, norm_weight, norm_bias, eps, q_weight, q_bias, trans_b, shape_param,
        k_weight=None, k_bias=None, v_weight=None, v_bias=None,
    ):
        if inputs.stride()[-1] != 1:
            inputs = inputs.contiguous()
        
        if q_bias != None:
            if isinstance(q_bias, int):
                dim = q_weight.shape[1] if trans_b == False else q_weight.shape[0]
                q_bias = torch.tensor(
                    [q_bias] * dim, device=inputs.device, dtype=inputs.dtype
                )
            q_bias_shape = q_bias.shape
            if len(q_bias_shape) == 2 and bias_shape[0] == 1:
                q_bias = q_bias.view(-1)
        if trans_b == False:
            q_weight = q_weight.transpose(0, 1).contiguous()

        output = torch_mlu_ops.fused_norm_attention_project(
            inputs,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            norm_weight,
            norm_bias,
            eps,
            #out_layout,
            #head_size,
            #norm_out,
        )
        if shape_param:
            output = output.view(shape_param)
        return output

def _is_layernorm_mm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if node.target != "mlu_tmo_fused_matmul_1_replacement" and \
       node.target != "mlu_tmo_fused_matmul_2_replacement":
        return False, ()
    _to_copy_node = node.args[0]
    if not check_copy(_to_copy_node):
        return False, ()

    layernorm_node = _to_copy_node.args[0]
    if layernorm_node.target != "layer_norm_op":
        return False, ()

    inputs = layernorm_node.args[0]
    norm_weight = layernorm_node.args[1]
    norm_bias = layernorm_node.args[2]
    eps = layernorm_node.args[3]

    input_shape = node.args[1]
    q_weight = node.args[2]
    q_weight_shape = node.args[3]
    trans_b = node.args[4]
    q_bias = node.args[5]
    shape_param = node.args[6]

    if q_bias != None:
        q_bias_shape = q_bias.meta["tensor_meta"].shape
        if len(q_bias_shape) == 2 and q_bias_shape[0] > 1:
            return False, ()
    
    return True, (inputs, norm_weight, norm_bias, eps, q_weight, q_bias, trans_b, shape_param)
    

class FusedLayernormMM(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule(
            "mlu_tmo_fused_norm_mm_replacement", FusedLayernormMMReplacement()
        )

        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_layernorm_mm(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_tmo_fused_norm_mm_replacement",
                        args=(params),
                )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True
        graph_module.graph.lint()
        graph_module.recompile()
        print(graph_module.graph)
        return is_modified
