import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from typing import List, Tuple

from xpu_graph import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_copy,
    check_add_op,
    check_bmm_op,
    check_softmax_op,
    get_actual_node,
    get_shape,
    check_div_or_mul_op,
    check_sub_or_add_op,
)


@torch.library.custom_op("torch_mlu::tmo_fa_forward", mutates_args=())
def tmo_fa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scale_factor: torch.Tensor,
    is_division: bool,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if query.dtype != output_dtype:
        query = query.to(output_dtype)
    if key.dtype != output_dtype:
        key = key.to(output_dtype)
    if value.dtype != output_dtype:
        value = value.to(output_dtype)

    if len(query.shape) == 4:
        batch_size, num_heads, sequence_len, head_dim = query.shape
    else:
        num_heads, sequence_len, head_dim = query.shape
        batch_size = 1
    key_sequence_len = key.shape[-2]

    if attention_mask != None:
        if attention_mask.dtype != output_dtype:
            attention_mask = attention_mask.to(output_dtype)
        if is_add == False:
            attention_mask = torch.neg(attention_mask)
        attention_mask = torch.broadcast_to(
            attention_mask, (batch_size, num_heads, sequence_len, key_sequence_len)
        ).contiguous()

    scale_factor = scale_factor.item()
    softmax_scale = 1.0 / scale_factor if is_division else scale_factor

    if num_heads <= 128:
        if len(query.shape) == 4:
            query = query.transpose(2, 1)
            key = key.transpose(2, 1)
            value = value.transpose(2, 1)
        else:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        key = key.contiguous()
        output = torch_mlu_ops.flash_attention(
            query,
            key,
            value,
            None,
            torch.tensor([0, sequence_len], dtype=torch.int32, device="mlu"),
            torch.tensor([0, key_sequence_len], dtype=torch.int32, device="mlu"),
            None,
            attention_mask,
            sequence_len,
            key_sequence_len,
            softmax_scale,
            False,
        )
        output = output.reshape(-1, sequence_len, num_heads, head_dim).transpose(1, 2)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output.view(output_shape)
    else:
        qk = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
        if attention_mask != None:
            if len(attention_mask.shape) == 4:
                qk = qk.view(batch_size, num_heads, sequence_len, key_sequence_len)
            qk += attention_mask
        qk = qk.softmax(dim=-1)
        qk = qk.view(-1, sequence_len, key_sequence_len)
        output = torch.bmm(qk, value)
        if output.dtype != output_dtype:
            output = output.to(output_dtype)
        return output.view(output_shape)


@tmo_fa_forward.register_fake
def tmo_fa_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scale_factor: torch.Tensor,
    is_division: bool,
    is_add: bool,
    output_shape: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    output_tensor = torch.empty(
        output_shape,
        dtype=output_dtype,
        device=query.device,
    )
    return output_tensor


class FlashAttentionReplacement(nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        scale_params,
        add_params,
        output_shape,
        output_dtype,
    ):
        scale_factor, is_division = scale_params
        attention_mask, is_add = add_params

        if isinstance(scale_factor, float):
            scale_factor = torch.tensor(scale_factor)

        return tmo_fa_forward(
            query,
            key,
            value,
            attention_mask,
            scale_factor,
            is_division,
            is_add,
            output_shape,
            output_dtype,
        )


class FlashAttentionWithTranspose(nn.Module):
    def __init__(self):
        super().__init__()
        self.flash_attention = FlashAttentionReplacement()

    def forward(
        self,
        query,
        key_transposed,
        value,
        scale_params,
        add_params,
        output_shape,
        output_dtype,
    ):
        key = key_transposed.transpose(-1, -2)
        return self.flash_attention(
            query, key, value, scale_params, add_params, output_shape, output_dtype
        )


def validate_transpose_operation(key_transpose):
    if not isinstance(key_transpose, fx.Node) or key_transpose.op != "call_function":
        return False

    if key_transpose.target != torch.ops.aten.transpose.int:
        return False

    dim1, dim2 = key_transpose.args[1:]
    valid_dimensions = [(-2, -1), (-1, -2), (1, 2), (2, 3)]
    return (dim1, dim2) in valid_dimensions


def _is_fa(node: fx.Node):
    if node.target != "fused_bmm":
        return False, []
    softmax_node = get_actual_node(node, 0)
    if not check_softmax_op(softmax_node):
        return False, []

    bmm_1_node = get_actual_node(softmax_node, 0)

    # (optional) find add
    add_params = (None, False)
    is_scale_op, addinput1, params = check_sub_or_add_op(bmm_1_node)
    if is_scale_op:
        add_params = params
        bmm_1_node = addinput1

    # (optional) find div or mul
    scale_params = (1.0, False)
    is_scale_op, div_input_node, params = check_div_or_mul_op(bmm_1_node)
    if is_scale_op:
        scale_params = params
        bmm_1_node = div_input_node

    if bmm_1_node.target != "fused_bmm":
        if bmm_1_node.target != "fused_baddbmm":
            return False, []
        if add_params[0] != None:
            logger.warning("Flash attention pass: Too many add operations")
            return False, []
        add_params[0] = bmm_1_node.args[2]
        add_params[1] = True

    return True, [
        bmm_1_node.args[0],
        bmm_1_node.args[1],
        node.args[1],
        scale_params,
        add_params,
        list(node.args[4]),
        node.args[5],
    ]


class FusedFlashAttention(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule):
        graph_module.add_submodule("flash_attn_base", FlashAttentionReplacement())
        graph_module.add_submodule(
            "flash_attn_transpose", FlashAttentionWithTranspose()
        )
        modified = False
        for node in reversed(graph_module.graph.nodes):
            matched, fa_param = _is_fa(node)
            if not matched:
                continue
            with graph_module.graph.inserting_before(node):
                fused = graph_module.graph.call_module(
                    "flash_attn_transpose",
                    args=tuple(fa_param),
                )

            key_trans = fa_param[1]
            if validate_transpose_operation(key_trans):
                fa_param[1] = key_trans.args[0]
                with graph_module.graph.inserting_before(node):
                    fused = graph_module.graph.call_module(
                        "flash_attn_base",
                        args=tuple(fa_param),
                    )
            node.replace_all_uses_with(fused)
            modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return modified
