"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""
import torch
import torch_mlu
import triton
import triton.language as tl


@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    log2e = -1.442695041
    y = 1.0 / (1.0 + tl.exp2(x_fp32 * log2e)) * x_fp32
    return y


@triton.jit
def _attn_fwd_inner(
    off_z,
    acc,
    q,  #
    BIAS,
    K_block_ptr,
    V_block_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EXPANDED_BIAS: tl.constexpr,
):
    # range of values handled by this stage
    if CAUSAL:
        lo, hi = 0, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_BIAS:
            bias = None
            if EXPANDED_BIAS:
                if IS_DIVISIBLE:
                    bias = tl.load(BIAS + off_z * N_CTX * N_CTX + offs_m[:, None] * N_CTX + start_n + offs_n)
                else:
                    bias = tl.load(
                        BIAS + off_z * N_CTX * N_CTX + offs_m[:, None] * N_CTX + start_n + offs_n,
                        mask=(offs_m < N_CTX)[:, None] & (offs_n < N_CTX)[None, :],
                        other=0.0,
                    )
            else:
                if IS_DIVISIBLE:
                    bias = tl.load(BIAS + off_z * N_CTX + start_n + offs_n)[None, :]
                else:
                    bias = tl.load(
                        BIAS + off_z * N_CTX + start_n + offs_n,
                        offs_n < N_CTX,
                        other=0.0,
                    )[None, :]
            qk += tl.dot(q, k)
            qk = qk * qk_scale
            qk = qk * (bias != 0).to(qk.dtype)
            # qk = qk + bias
        else:
            qk += tl.dot(q, k)
            qk = qk * qk_scale
        if CAUSAL and start_n >= start_m * BLOCK_M:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
        p = silu_forward(qk)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk_wram = p.to(v.type.element_ty)
        qkv = tl.dot(qk_wram, v)
        # -- update output accumulator --
        acc += qkv

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc


# We don't run auto-tuning everytime to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# @triton.autotune(
#   configs=[
#       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=1),
#       #triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
#       #triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=1),
#   ],
#   key=['N_CTX'],
# )


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    BIAS,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    IS_DIVISIBLE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EXPANDED_BIAS: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)
    context_num = triton.cdiv(N_CTX, BLOCK_M)
    total_heads = Z * H * context_num
    deal_num_per_core = total_heads // program_dim
    extra_deal_core_num = total_heads - deal_num_per_core * program_dim
    deal_num_per_core += 1
    core_head_begin = program_id * deal_num_per_core
    if program_id >= extra_deal_core_num:
        deal_num_per_core -= 1
        core_head_begin = program_id * deal_num_per_core + extra_deal_core_num
    if deal_num_per_core <= 0:
        return
    head_begin = core_head_begin
    head_end = head_begin + deal_num_per_core

    for head_idx in range(head_begin, head_end):
        start_m = head_idx % context_num
        off_hz = head_idx // context_num
        off_z = off_hz // H
        off_h = off_hz % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        # qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = _attn_fwd_inner(
            off_z,
            acc,
            q,
            BIAS,
            K_block_ptr,
            V_block_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,  #
            CAUSAL,
            offs_m,
            offs_n,
            N_CTX,  #
            IS_DIVISIBLE,
            HAS_BIAS,
            EXPANDED_BIAS,
        )

        if IS_DIVISIBLE:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, causal, sm_scale, has_bias, expanded_bias):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 128
        if Lq == 256:
            BLOCK_N = 64

        num_warps = 1
        num_stages = 5
        processor_count = torch.mlu.get_device_properties(torch.mlu.current_device()).multi_processor_count
        grid = (processor_count, 1, 1)

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        N_CTX = q.shape[2]
        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True

        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            bias,
            o,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            q.shape[0],
            q.shape[1],  #
            N_CTX,  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk,  #
            CAUSAL=causal,  #
            IS_DIVISIBLE=IS_DIVISIBLE,  #
            HAS_BIAS=has_bias,
            EXPANDED_BIAS=expanded_bias,
            num_warps=num_warps,  #
            num_stages=num_stages,  #
        )

        return o


attention = _attention.apply
