from typing import Optional

import torch


@torch.library.custom_op("xpu_graph::mlu_fa_wrapped_scale", mutates_args=())
def mlu_fa_wrapped_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],
    cu_seq_lens_kv: Optional[torch.Tensor],
    alibi_slope: Optional[torch.Tensor],
    attn_bias: Optional[torch.Tensor],
    max_seq_len_q: int,
    max_seq_len_kv: int,
    softmax_scale: torch.Tensor,
    is_causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    compute_dtype: torch.dtype = torch.float,
    return_lse: bool = False,
    block_tables: Optional[torch.Tensor] = None,
    k_quant_scale: Optional[torch.Tensor] = None,
    v_quant_scale: Optional[torch.Tensor] = None,
    q_quant_scale: Optional[torch.Tensor] = None,
    out_quant_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
) -> torch.Tensor:
    softmax_scale = softmax_scale.item()
    import torch_mlu_ops

    return torch_mlu_ops.flash_attention(
        q,
        k,
        v,
        out,
        cu_seq_lens_q,
        cu_seq_lens_kv,
        alibi_slope,
        attn_bias,
        max_seq_len_q,
        max_seq_len_kv,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        compute_dtype,
        return_lse,
        block_tables,
        k_quant_scale,
        v_quant_scale,
        q_quant_scale,
        out_quant_scale,
        out_dtype,
    )


@torch.library.register_fake("xpu_graph::mlu_fa_wrapped_scale")
def _(
    q,
    k,
    v,
    out,
    cu_seq_lens_q,
    cu_seq_lens_kv,
    alibi_slope,
    attn_bias,
    max_seq_len_q,
    max_seq_len_kv,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    compute_dtype,
    return_lse,
    block_tables,
    k_quant_scale,
    v_quant_scale,
    q_quant_scale,
    out_quant_scale,
    out_dtype,
):
    softmax_scale = 1.0
    import torch_mlu_ops

    return torch_mlu_ops.flash_attention(
        q,
        k,
        v,
        out,
        cu_seq_lens_q,
        cu_seq_lens_kv,
        alibi_slope,
        attn_bias,
        max_seq_len_q,
        max_seq_len_kv,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        compute_dtype,
        return_lse,
        block_tables,
        k_quant_scale,
        v_quant_scale,
        q_quant_scale,
        out_quant_scale,
        out_dtype,
    )


class FlashAttentionModule(torch.nn.Module):
    def forward(self, q, k, v, attn_mask, scale):
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        # Note: convert shape from [B,N,S,D] to [B,S,N,D]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        k = k.contiguous()
        if isinstance(scale, torch.Tensor):
            o = torch.ops.xpu_graph.mlu_fa_wrapped_scale(
                q,
                k,
                v,
                None,
                torch.tensor([0, q_len], dtype=torch.int32, device="mlu"),
                torch.tensor([0, kv_len], dtype=torch.int32, device="mlu"),
                None,
                attn_mask,
                q_len,
                kv_len,
                scale,
                False,
            )
        else:
            import torch_mlu_ops

            o = torch_mlu_ops.flash_attention(
                q,
                k,
                v,
                None,
                torch.tensor([0, q_len], dtype=torch.int32, device="mlu"),
                torch.tensor([0, kv_len], dtype=torch.int32, device="mlu"),
                None,
                attn_mask,
                q_len,
                kv_len,
                scale,
                False,
            )
        # Note: convert shape back from [B,S,N,D] to [B,N,S,D]
        o = o.transpose(-2, -3)
        return o


def can_fuse_fa(q, k, v, attn_mask, scale):
    num_heads = q.shape[-3]
    # Magic number
    return num_heads <= 128
