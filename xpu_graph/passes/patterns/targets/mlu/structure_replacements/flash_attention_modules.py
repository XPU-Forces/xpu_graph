import torch


class FlashAttentionModule(torch.nn.Module):
    def forward(self, q, k, v, attn_mask, scale):
        q_len = q.shape[-2]
        kv_len = k.shape[-2]
        # Note: convert shape from [B,N,S,D] to [B,S,N,D]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        k = k.contiguous()
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
