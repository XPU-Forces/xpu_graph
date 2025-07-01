import math

import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def _sfdp_pattern_1(query, key, value, inv_scale):
    # query:bsz, self.num_heads, q_len, head_dim
    return torch.matmul(query, key.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(value)


def _sfdp_pattern_1_1(query, key, value, inv_scale):
    # query:bsz, self.num_heads, q_len, head_dim
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(torch.clamp(torch.tensor([inv_scale], dtype=query.dtype, device=query.device), 0, 20))
        .softmax(dim=-1)
        .matmul(value)
    )


def _sfdp_pattern_2(query, key, value, scale_factor):
    return torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1).matmul(value)


def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p=0.0):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p=0.0):
    return torch.nn.functional.dropout(
        torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


def _sfdp_pattern_5_1(query, key, value):
    attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))), dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def _sfdp_pattern_5(query, key, value, attn_mask):
    attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p)
    return attn_weight @ value


def _sfdp_pattern_6(query, key, value, attn_mask, dropout_p=0.0):
    attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


def _sfdp_pattern_6_1(query, key, value, attn_mask, dropout_p=0.0):
    attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) - attn_mask, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    return attn_weight @ value


def _sfdp_pattern_7(query, key, value, dropout_p=0.0):
    # in real workloads inputs to matmul are permuted
    # causing matmul to expand to a series of expand and clone calls
    # we want the same to happen during pattern tracing
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_pattern_8(query, key, value):
    # no dropout version of pattern 7
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_pattern_9(query, key, value, dropout_p=0.0):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / math.sqrt(q.size(-1))
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_pattern_10(query, key, value):
    # no dropout version of 9
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    q = q / math.sqrt(q.size(-1))
    div = q @ k.transpose(-2, -1)
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v


def _sfdp_pattern_11(query, key, value, inv_scale):
    # Mainly for huggingface models
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.matmul(q, k.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(v)


def _sfdp_pattern_12(query, key, value, inv_scale_factor, dropout_p=0.0):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return torch.nn.functional.dropout(
        torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(v)


def _sfdp_pattern_13(query, key, value, dropout_p=0.0):
    attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    return torch.bmm(attn_weight, value)


def _sfdp_pattern_14(query, key, value, inv_scale, attn_mask):
    # for BertLarge
    # Permutations are needed to create clones in graph.
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    return (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1).matmul(v)


# TODO
def _sfdp_pattern_15(query, key, value, inv_scale, attn_mask):
    # for DistilBert
    # Permutations are needed to create clones in graph.
    # Ref: https://github.com/pytorch/pytorch/issues/119911
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    bs = q.size(0)
    k_len = k.size(-2)
    scores = q @ k.transpose(-2, -1)
    scores = scores.div(inv_scale)
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v


def _sfdp_pattern_16(query, key, value, inv_scale, attn_mask, dropout_p=0.0):
    # for BertLarge with dropout
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    return (
        torch.nn.functional.dropout(
            (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(dim=-1),
            dropout_p,
        )
        .to(dtype=query.dtype)
        .matmul(v)
    )


def _sfdp_pattern_17(query, key, value, attn_mask, inv_scale, dropout_p):
    # for DistilBert with dropout
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    bs = q.size(0)
    k_len = k.size(-2)
    scores = q @ k.transpose(-2, -1)
    scores = scores.div(inv_scale)
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return torch.nn.functional.dropout(torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p) @ v


def _sfdp_pattern_18(query, key, value, causal_mask, dropout_p):
    # for hf_GPT2 with dropout (introduces clone node) for inference
    # it also returns permuted key & value
    query = query.permute([0, 2, 1, 3])
    key = key.permute([0, 2, 1, 3])
    value = value.permute([0, 2, 1, 3])
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    inv_scale = torch.full(
        [],
        value.size(-1) ** 0.5,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    attn_weights = attn_weights.div(inv_scale)
    causal_mask_value = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
    attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
    return (
        (torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(value)),
        key,
        value,
    )


def _sfdp_pattern_19(query, key, value, causal_mask, attn_mask, dropout_p):
    # for token-classification+gpt2 / text-generation+gpt2
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    inv_scale = torch.full(
        [],
        value.size(-1) ** 0.5,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    attn_weights = attn_weights.div(inv_scale)
    causal_mask_value = torch.full((), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device)
    attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
    attn_weights = attn_weights + attn_mask
    attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
    return torch.nn.functional.dropout(attn_weights, dropout_p).matmul(value)


def _sfdp_pattern_transformer_1(query, key, value):
    # llama
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=False)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def _sfdp_pattern_transformer_2(query, key, value, attention_mask):
    # llama/qwen/mixtral
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=False)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def _sfdp_pattern_transformer_3(query, key, value, attention_mask):
    # falcon
    attention_scores = query @ key.transpose(-1, -2)
    attention_scores /= math.sqrt(query.size(-1))

    attention_scores = torch.nn.functional.softmax(attention_scores + attention_mask, dim=-1, dtype=query.dtype)
    # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
    attn_output = attention_scores @ value
    return attn_output


def fa_test(xpu_graph_backend, func):
    head_size = 64
    seq_q, seq_k = 38, 38
    head_num_q, head_num_k = 32, 32
    dtype = torch.half
    batch = 1
    softmax_scale = None
    if func in [
        _sfdp_pattern_1,
        _sfdp_pattern_1_1,
        _sfdp_pattern_3,
        _sfdp_pattern_11,
        _sfdp_pattern_12,
        _sfdp_pattern_14,
        _sfdp_pattern_15,
        _sfdp_pattern_16,
    ]:
        softmax_scale = 1 / math.sqrt(head_size)
    elif func in [_sfdp_pattern_2, _sfdp_pattern_4]:
        softmax_scale = math.sqrt(head_size)

    if func in [
        _sfdp_pattern_1,
        _sfdp_pattern_1_1,
        _sfdp_pattern_2,
        _sfdp_pattern_3,
        _sfdp_pattern_4,
        _sfdp_pattern_5,
        _sfdp_pattern_5_1,
        _sfdp_pattern_6,
        _sfdp_pattern_6_1,
        _sfdp_pattern_13,
        _sfdp_pattern_transformer_1,
        _sfdp_pattern_transformer_2,
        _sfdp_pattern_transformer_3,
    ]:
        q = torch.randn(batch, head_num_q, seq_q, head_size, dtype=dtype, device="mlu")
        k = torch.randn(batch, head_num_k, seq_k, head_size, dtype=dtype, device="mlu")
        v = torch.randn(batch, head_num_k, seq_k, head_size, dtype=dtype, device="mlu")
    else:  # have trans
        q = torch.randn(batch, seq_q, head_num_q, head_size, dtype=dtype, device="mlu")
        k = torch.randn(batch, seq_k, head_num_k, head_size, dtype=dtype, device="mlu")
        v = torch.randn(batch, seq_k, head_num_k, head_size, dtype=dtype, device="mlu")

    if func in [_sfdp_pattern_13]:
        # need 3d
        q = q.reshape(-1, *q.shape[2:])
        k = k.reshape(-1, *k.shape[2:])
        v = v.reshape(-1, *v.shape[2:])

    args = (q, k, v, 1 / softmax_scale) if softmax_scale else (q, k, v)

    if func in [
        _sfdp_pattern_5,
        _sfdp_pattern_6,
        _sfdp_pattern_6_1,
        _sfdp_pattern_14,
        _sfdp_pattern_16,
        _sfdp_pattern_transformer_2,
        _sfdp_pattern_transformer_3,
    ]:
        attn_bias = torch.randn((batch, head_num_q, seq_q, seq_k), dtype=dtype, device="mlu")
        args += (attn_bias,)

    res1 = func(*args)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res = compiled(*args)

    assertTensorsEqual(res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True)


class TestFA:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            _sfdp_pattern_transformer_1,
            _sfdp_pattern_transformer_2,
            _sfdp_pattern_transformer_3,
            _sfdp_pattern_1,
            _sfdp_pattern_1_1,
            _sfdp_pattern_2,
            _sfdp_pattern_3,
            _sfdp_pattern_4,
            _sfdp_pattern_5,
            _sfdp_pattern_5_1,
            _sfdp_pattern_6,
            _sfdp_pattern_6_1,
            _sfdp_pattern_7,
            _sfdp_pattern_8,
            _sfdp_pattern_9,
            _sfdp_pattern_10,
            _sfdp_pattern_11,
            _sfdp_pattern_12,
            _sfdp_pattern_13,
        ],
    )
    def test_sfdp_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            fa_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedFlashAttention changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, debug=False, opt_level=OptLevel.level2)
    fa_test(xpu_graph_backend, _sfdp_pattern_1)
    fa_test(xpu_graph_backend, _sfdp_pattern_1_1)
    fa_test(xpu_graph_backend, _sfdp_pattern_2)
    fa_test(xpu_graph_backend, _sfdp_pattern_3)
    fa_test(xpu_graph_backend, _sfdp_pattern_4)
    fa_test(xpu_graph_backend, _sfdp_pattern_5)
    fa_test(xpu_graph_backend, _sfdp_pattern_6)
    fa_test(xpu_graph_backend, _sfdp_pattern_7)
    fa_test(xpu_graph_backend, _sfdp_pattern_8)
    fa_test(xpu_graph_backend, _sfdp_pattern_9)
    fa_test(xpu_graph_backend, _sfdp_pattern_10)
    fa_test(xpu_graph_backend, _sfdp_pattern_11)
    fa_test(xpu_graph_backend, _sfdp_pattern_12)
    fa_test(xpu_graph_backend, _sfdp_pattern_13)
    fa_test(xpu_graph_backend, _sfdp_pattern_5_1)
    fa_test(xpu_graph_backend, _sfdp_pattern_transformer_1)
    fa_test(xpu_graph_backend, _sfdp_pattern_transformer_2)
    fa_test(xpu_graph_backend, _sfdp_pattern_transformer_3)
    fa_test(xpu_graph_backend, _sfdp_pattern_6_1)
