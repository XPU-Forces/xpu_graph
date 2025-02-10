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
    check_add_op,
    check_norm_op,
    check_view,
    check_getitem_op,
)


class FusedNormReplacement(nn.Module):
    def forward(
        self,
        input,
        residual,
        weight,
        bias,
        residual_bias,
        eps,
        store_output_before_norm,
        norm_type,
    ):
        if bias is None:
            bias = torch.zeros_like(weight)
        if store_output_before_norm:
            if norm_type == "layer_norm":
                output, residual = torch_mlu_ops.fused_layer_norm(
                    input,
                    residual,
                    weight,
                    bias,
                    None,
                    eps,
                    True,
                )
            else:
                output, residual = torch_mlu_ops.fused_rms_norm(
                    input,
                    residual,
                    weight,
                    bias,
                    None,
                    eps,
                    True,
                )
            return output, residual
        else:
            if norm_type == "layer_norm":
                output = torch_mlu_ops.fused_layer_norm(
                    input,
                    residual,
                    weight,
                    bias,
                    None,
                    eps,
                    False,
                )
            else:
                output = torch_mlu_ops.fused_rms_norm(
                    input,
                    residual,
                    weight,
                    bias,
                    None,
                    eps,
                    False,
                )
            return output


def _is_add_norm(node: fx.Node, match_str: str):
    layernorm_node = node
    is_norm, norm_type = check_norm_op(layernorm_node)
    if not is_norm:
        return False, ()

    if norm_type != match_str:
        return False, ()

    if match_str == "layer_norm":
        add_node = layernorm_node.args[0]
        if not check_add_op(add_node):
            return False, ()
        # normalized_shape = layernorm_node.args[1]
        weight = layernorm_node.args[2]
        if weight is None:
            return False, ()
        bias = layernorm_node.args[3]
        eps = layernorm_node.args[4]
    else:
        add_node = layernorm_node.args[0]
        if not check_add_op(add_node):
            return False, ()
        weight = layernorm_node.args[1]
        eps = layernorm_node.args[2]
        bias = None

    inputs = add_node.args[0]
    residual = add_node.args[1]

    inputs_dtype = inputs.meta["tensor_meta"].dtype
    residual_dtype = residual.meta["tensor_meta"].dtype
    weight_dtype = weight.meta["tensor_meta"].dtype
    if inputs_dtype != residual_dtype:
        return False, ()

    if inputs_dtype != weight_dtype:
        return False, ()

    store_output_before_norm = False

    if len(add_node.users) > 1:
        store_output_before_norm = True

    return True, (
        layernorm_node,
        add_node,
        inputs,
        residual,
        weight,
        bias,
        None,
        eps,
        store_output_before_norm,
    )


def _is_add_layernorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    if not check_getitem_op(node):
        return False, ()
    return _is_add_norm(node.args[0], "layer_norm")


def _is_add_rmsnorm(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    return _is_add_norm(node, "rms_norm")


def node_replacement(
    node: fx.Node, graph_module: fx.GraphModule, match_str: str, params
):
    if params[-1]:
        with graph_module.graph.inserting_before(node):
            new_node = graph_module.graph.call_module(
                "mlu_tmo_fused_norm_replacement",
                args=(params[2:] + (match_str,)),
            )
            normalized_node = graph_module.graph.call_function(
                operator.getitem,
                args=(new_node, 0),
                kwargs={},
            )
            residual_node = graph_module.graph.call_function(
                operator.getitem,
                args=(new_node, 1),
                kwargs={},
            )

        node.replace_all_uses_with(normalized_node)
        params[1].replace_all_uses_with(residual_node)
        graph_module.graph.erase_node(node)
        graph_module.graph.erase_node(params[1])
        graph_module.graph.erase_node(params[0])
    else:
        with graph_module.graph.inserting_before(node):
            new_node = graph_module.graph.call_module(
                "mlu_tmo_fused_norm_replacement",
                args=(params[2:] + (match_str,)),
            )
        node.replace_all_uses_with(new_node)
        graph_module.graph.erase_node(node)


class FusedAddLayerNorm(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        graph_module.add_submodule(
            "mlu_tmo_fused_norm_replacement", FusedNormReplacement()
        )

        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_add_layernorm(node)
            if is_match:
                node_replacement(node, graph_module, "layer_norm", params)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified

class FusedAddRMSNorm(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        graph_module.add_submodule(
            "mlu_tmo_fused_norm_replacement", FusedNormReplacement()
        )

        for node in reversed(graph_module.graph.nodes):
            is_match, params = _is_add_rmsnorm(node)
            if is_match:
                node_replacement(node, graph_module, "rms_norm", params)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
