import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from typing import List, Tuple

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    get_actual_node,
    get_shape,
    check_div_or_mul_op,
    check_mask_fill_op,
    check_eq_op,
    check_repeat_op,
    check_unsqueeze_op,
    check_act_op,
)

from .triton_kernel.linear_attention_kernel import attention


def naive(q, k, v, bias, causal, sm_scale, has_bias):
    N_CTX = q.shape[-2]
    n_head = q.shape[1]
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="mlu"))
    p = torch.matmul(q, k.transpose(2, 3)).float() * sm_scale
    if has_bias:
        p = p * (bias != 0).to(p.dtype)
    if causal:
        p[:, :, M == 0] = float("-1e6")
    p = torch.nn.functional.silu(p).to(q.dtype)
    ref_out = torch.matmul(p, v)
    return ref_out


class LinearAttentionReplacement(nn.Module):
    def forward(self, query, key, value, bias, causal, scale_params, has_bias):
        key = key.transpose(2, 3)
        sm_scale, is_div = scale_params
        if is_div:
            sm_scale = 1.0 / sm_scale

        # output = naive(
        output = attention(query, key, value, bias, causal, sm_scale, has_bias)
        return output


def _is_bias(silu_node: fx.Node):
    mask_fill_node = silu_node.args[0]
    if not check_mask_fill_op(mask_fill_node):
        return False, []
    bmm_node = mask_fill_node.args[0]
    eq_node = mask_fill_node.args[1]
    if not check_eq_op(eq_node):
        return False, []
    repeat_node = eq_node.args[0]
    if not check_repeat_op(repeat_node):
        return False, []
    unsqueeze_node = repeat_node.args[0]
    if not check_unsqueeze_op(unsqueeze_node):
        return False, []
    bias_node = unsqueeze_node.args[0]
    return True, [bmm_node, bias_node]


def _is_liear(node: fx.Node):
    if node.target != "fused_bmm":
        return False, []
    silu_node = get_actual_node(node, 0)
    is_act, act_str_ = check_act_op(silu_node)
    if not is_act:
        return False, []
    if act_str_ != "silu":
        return False, []

    bmm_1_node = get_actual_node(silu_node, 0)

    # (optional) find bias
    bias = None
    has_bias, params = _is_bias(silu_node)
    if has_bias:
        bmm_1_node, bias = params

    # TODO(jyj) find causal
    causal = False

    # (optional) find mul
    scale_params = (1.0, False)
    is_scale_op, div_input_node, params = check_div_or_mul_op(bmm_1_node)
    if is_scale_op:
        scale_params = params
        bmm_1_node = div_input_node

    if bmm_1_node.target != "fused_bmm":
        return False, []

    query = bmm_1_node.args[0]
    key = bmm_1_node.args[1]
    value = node.args[1]

    return True, [query, key, value, bias, causal, scale_params, has_bias]


class FusedLinearAttention(Pattern):
    def process(self, graph_module: fx.GraphModule):
        modified = False
        graph_module.add_submodule("linear_attention", LinearAttentionReplacement())
        for node in reversed(graph_module.graph.nodes):
            matched, linear_param = _is_liear(node)
            if not matched:
                continue
            with graph_module.graph.inserting_before(node):
                fused = graph_module.graph.call_module(
                    "linear_attention",
                    args=tuple(linear_param),
                )
            node.replace_all_uses_with(fused)
            modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return modified
