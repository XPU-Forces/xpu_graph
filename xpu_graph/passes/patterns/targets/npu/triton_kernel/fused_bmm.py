import math
import torch
import torch_npu
import triton
import triton.language as tl
from typing import List

bmm_strategy = {
    # (B, M, N, K):[(BLOCK_M, BLOCK_N, BLOCK_K)]
    # (86, 1392, 32, 32):["bmm_01", (128, 32, 32)],
    (172, 48, 176, 32):["bmm_02", (48, 88, 32)],
    (172, 48, 128, 176):["bmm_02", (48, 128, 176)],
    (172, 2, 48, 32):["bmm_02", (2, 48, 32)],
    (172, 2, 128, 48):["bmm_02", (2, 128, 48)],
    (86, 64, 32, 128):["bmm_01", (64, 32, 128)],
    (86, 64, 32, 32):["bmm_01", (64, 32, 32)],
    (86, 32, 32, 128):["bmm_01", (32, 32, 128)],
    (86, 80, 32, 64):["bmm_01", (80, 32, 64)],
    (86, 47, 47, 8):["bmm_01", (47, 47, 8)],
    (86, 47, 8, 8):["bmm_01", (47, 8, 8)],
    (86, 1, 256, 1):["bmm_01", (1, 256, 1)],
}

dtype_dict1 = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}

@triton.jit
def npu_triton_bmm_splitM(
        a_ptr, b_ptr, c_ptr,
        kernel_BS: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_ab: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bb: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cb: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for bs in range(0, kernel_BS):
        b_ptrs = b_ptr + (pid * kernel_BS + bs) * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs)

        for im in range(0, M, BLOCK_M):
            a_ptrs = a_ptr + (pid * kernel_BS + bs) * stride_ab + (im * BLOCK_M + offs_m[:, None]) * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(a_ptrs)
            c = tl.dot(a, b)
            c_ptrs = c_ptr + (pid * kernel_BS + bs) * stride_cb + (im * BLOCK_M + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
            c_mask = ((im * BLOCK_M + offs_m[:, None]) < M) & (offs_n[None, :] * stride_cn < N)
            tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def npu_triton_bmm_splitN(
        # 数据指针
        a_ptr, b_ptr, c_ptr,
        # 张量维度信息
        kernel_BS: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_ab: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bb: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cb: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for bs in range(0, kernel_BS):
        a_ptrs = a_ptr + (pid * kernel_BS + bs) * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs)
        for _in in range(0, N, BLOCK_N):
            b_ptrs = b_ptr + (pid * kernel_BS + bs) * stride_bb + (offs_k[:, None]) * stride_bk + (_in * BLOCK_N + offs_n[None, :]) * stride_bn
            b = tl.load(b_ptrs)
            c = tl.dot(a, b)
            c_ptrs = c_ptr + (pid * kernel_BS + bs) * stride_cb + offs_m[:, None] * stride_cm + (_in * BLOCK_N + offs_n[None, :]) * stride_cn
            c_mask = (offs_m[:, None] < M) & ((_in * BLOCK_N + offs_n[None, :]) * stride_cn < N)
            tl.store(c_ptrs, c, mask=c_mask)

def bmm_splitM(a, b, block_m, block_n, block_k):
    B, M, K = a.shape
    B, K, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    AICoreNum = 25
    kernel_BS = math.ceil(B / AICoreNum)  # ceil(86 / 20) = ceil(4.3) = 5
    coreNum = math.ceil(B / kernel_BS)  # ceil(86 / 5) = ceil(17.5) = 18
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, block_n, block_k

    grid = (coreNum,)
    npu_triton_bmm_splitM[grid](
        a, b, c,
        kernel_BS, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

def bmm_splitN(a, b, block_m, block_n, block_k):
    B, M, K = a.shape
    B, K, N = b.shape

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    AICoreNum = 25
    kernel_BS = math.ceil(B / AICoreNum)  # ceil(86 / 20) = ceil(4.3) = 5
    coreNum = math.ceil(B / kernel_BS)  # ceil(86 / 5) = ceil(17.5) = 18
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, block_n, block_k

    grid = (coreNum,)

    npu_triton_bmm_splitN[grid](
        a, b, c,
        kernel_BS, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta
npu_def.define("triton_bmm(Tensor a, Tensor b) -> (Tensor)")

def nearest_power_of_two(n: int) -> int:
    floor_pow = 1 << (n.bit_length() - 1)

    ceil_pow = floor_pow << 1
    
    if floor_pow == n:
        return n
    
    if n - floor_pow < ceil_pow - n:
        return floor_pow
    else:
        return ceil_pow
    
@impl(npu_lib, "triton_bmm")
def triton_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    # 根据b/m/n/k读取tiling策略
    B, M, K = a.shape
    B, K, N = b.shape
    c = None
    st = bmm_strategy.get((B, M, N, K), None)
    if st is not None:
        op, (BLOCK_M, BLOCK_N, BLOCK_K) = st
        if op == "bmm_01":
            c = bmm_splitM(a, b, BLOCK_M, BLOCK_N, BLOCK_K)
        elif op == "bmm_02":
            c = bmm_splitN(a, b, BLOCK_M, BLOCK_N, BLOCK_K)
        else:
            print("error!")
            exit(3)
    else:
        c = torch.bmm(a, b)
    return c

@impl(npu_meta, "triton_bmm")
def triton_bmm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
):
    device = a.device
    dtype = a.dtype
    B, M, K = a.shape
    B, K, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    return c
