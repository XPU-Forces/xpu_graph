import copy

import torch
import torch_mlu
import triton
import triton.language as tl

from . import libentry


def do_config_prune(configs, named_args, **kwargs):
    M = named_args["M"]
    K1 = named_args["K1"]
    N1 = named_args["N1"]
    N2 = named_args["N2"]

    pruned_configs = []
    for config in configs:
        pruned_configs.append(config)
        new_config = copy.deepcopy(config)
        if K1 < 256:
            new_config.kwargs["BLOCK_SIZE_K1"] = K1
        pruned_configs.append(new_config)
    return pruned_configs


configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_K1": BK1}, num_stages=3, num_warps=1)
    for BM in [32, 64, 96]
    for BK1 in [64, 128, 256]
]


@triton.jit
def relu(x):
    zero = 0.0
    zero = zero.to(x.dtype)
    return tl.maximum(x, zero)


@libentry.libentry()
@libentry.libtuner(
    configs=configs,
    prune_configs_by={"early_config_prune": do_config_prune},
    key=["M", "K1", "N1", "N2"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
        "EVEN_K1": lambda args: args["K1"] % args["BLOCK_SIZE_K1"] == 0,
    }
)
@triton.jit
def fused_serial_mm_2dot_kernel(
    d_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    bias1_ptr,
    bias2_ptr,
    M,
    K1: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ck: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    is_up_bias: tl.constexpr,
    is_down_bias: tl.constexpr,
    is_up_act: tl.constexpr,
    is_down_act: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K1: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_k1: tl.constexpr = tl.cdiv(K1, BLOCK_SIZE_K1)
    pid_m = pid

    m_num = M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0)
    if pid_m >= m_num:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < M
    offs_k1 = tl.arange(0, BLOCK_SIZE_K1)
    offs_n1 = tl.arange(0, N1)
    offs_n2 = tl.arange(0, N2)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k1[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k1[:, None] * stride_bk + offs_n1[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_n1[:, None] * stride_ck + offs_n2[None, :] * stride_cn)

    accumulator = tl.zeros((BLOCK_SIZE_M, N1), dtype=tl.float32)
    for k in tl.range(num_pid_k1):
        k_remaining = K1 - k * BLOCK_SIZE_K1
        if EVEN_M:
            if EVEN_K1:
                a = tl.load(a_ptrs, cache_modifier=".ca")
            else:
                a = tl.load(a_ptrs, mask=offs_k1[None, :] < k_remaining, other=0.0, cache_modifier=".ca")
        else:
            if EVEN_K1:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0, cache_modifier=".ca")
            else:
                a = tl.load(
                    a_ptrs, mask=mask_m[:, None] & (offs_k1[None, :] < k_remaining), other=0.0, cache_modifier=".ca"
                )
        b = tl.load(b_ptrs, mask=offs_k1[:, None] < k_remaining, cache_modifier=".ca")
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K1 * stride_ak
        b_ptrs += BLOCK_SIZE_K1 * stride_bk

    if is_up_bias:
        bias1_ptrs = bias1_ptr + offs_n1
        bias1 = tl.load(bias1_ptrs, cache_modifier=".ca")
        accumulator = accumulator.to(d_ptr.type.element_ty) + bias1
    if is_up_act:
        accumulator = relu(accumulator)

    c = tl.load(c_ptrs, cache_modifier=".ca")
    d = tl.dot(accumulator.to(d_ptr.type.element_ty), c, allow_tf32=False)

    if is_down_bias:
        bias2_ptrs = bias2_ptr + offs_n2
        bias2 = tl.load(bias2_ptrs, cache_modifier=".ca")
        d = d.to(d_ptr.type.element_ty) + bias2
    if is_down_act:
        d = relu(d)

    d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n2[None, :]
    if EVEN_M:
        tl.store(d_ptrs, d)
    else:
        tl.store(d_ptrs, d, mask=mask_m[:, None])


@torch.library.custom_op("torch_mlu_triton::fuse_serial_mm_2dot", mutates_args=(), device_types="mlu")
def fuse_serial_mm_2dot(
    a: torch.Tensor,
    b: torch.Tensor,
    bias1: torch.Tensor,
    c: torch.Tensor,
    bias2: torch.Tensor,
    is_up_bias: bool,
    is_down_bias: bool,
    is_up_act: bool,
    is_down_act: bool,
) -> torch.Tensor:
    d = torch.empty(
        [a.shape[0], c.shape[1]],
        dtype=a.dtype,
        device=a.device,
    )

    M, K1 = a.shape
    K1, N1 = b.shape
    N1, N2 = c.shape

    # TODO: ASSERT K1/N1/N2
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), )\
    align_dimx = 4
    grid = lambda META: ((triton.cdiv(M, META["BLOCK_SIZE_M"]) + align_dimx - 1) // align_dimx * align_dimx,)
    fused_serial_mm_2dot_kernel[grid](
        d,
        a,
        b,
        c,
        bias1,
        bias2,
        M,
        K1,
        N1,
        N2,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        d.stride(0),
        d.stride(1),
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    )
    return d


@fuse_serial_mm_2dot.register_fake
def fuse_serial_mm_2dot_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    bias1: torch.Tensor,
    c: torch.Tensor,
    bias2: torch.Tensor,
    is_up_bias: bool,
    is_down_bias: bool,
    is_up_act: bool,
    is_down_act: bool,
) -> torch.Tensor:
    # Check constraints.
    assert a.shape[1] == b.shape[0], "a, b Incompatible dimensions"
    assert b.shape[1] == c.shape[0], "b, c Incompatible dimensions"

    assert a.is_contiguous(), "a should be contiguous"
    assert b.is_contiguous(), "b should be contiguous"
    assert c.is_contiguous(), "c should be contiguous"

    assert a.dtype == b.dtype, "a, b should have the same dtype"
    assert b.dtype == c.dtype, "b, c should have the same dtype"

    M, K1 = a.shape
    K1, N1 = b.shape
    N1, N2 = c.shape
    size_of_dtype = 2
    if a.dtype == torch.float32:
        size_of_dtype = 4
    assert (
        K1 * N1 + N1 * N2 + N1 + N2 <= 512 / size_of_dtype * 1024
    ), "(K1 * N1 + N1 * N2 + N1 + N2) should be <= (512 / size_of_dtype * 1024)"

    d = torch.empty(
        [a.shape[0], c.shape[1]],
        dtype=a.dtype,
        device=a.device,
    )
    return d
