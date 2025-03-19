import math
import torch
import torch_npu
import triton
import triton.language as tl
from typing import List
from .mm_strategy import strategy

dtype_dict1 = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}

@triton.jit
def npu_triton_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def npu_triton_matmul_smallb_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        out_for_times: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b = tl.load(b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    for i in range(out_for_times):
        offs_am = (pid * out_for_times * BLOCK_SIZE_M + i * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        accumulator += tl.dot(a, b)

        offs_cm = pid * out_for_times * BLOCK_SIZE_M + i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta
npu_def.define("triton_matmul(Tensor a, Tensor b) -> (Tensor)")

def nearest_power_of_two(n: int) -> int:
    floor_pow = 1 << (n.bit_length() - 1)
    
    ceil_pow = floor_pow << 1
    
    if floor_pow == n:
        return n
    
    if n - floor_pow < ceil_pow - n:
        return floor_pow
    else:
        return ceil_pow

@triton.jit
def mm_01(a_ptr, b_ptr, c_ptr,
          M: tl.constexpr,
          N: tl.constexpr,
          K: tl.constexpr,
          stride_am: tl.constexpr,
          stride_ak: tl.constexpr,
          stride_bk: tl.constexpr,
          stride_bn: tl.constexpr,
          stride_cm: tl.constexpr,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_am = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    tl.store(c_ptrs, tl.dot(a, b), mask=c_mask)

def triton_matmul_001(a, b, block_m, block_n, block_k):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, block_n, block_k
    c = torch.empty((M, N), device="npu", dtype=torch.float16)
    grid = (math.ceil(M / BLOCK_M),)
    mm_01[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), BLOCK_M, BLOCK_N, BLOCK_K)

    return c

@triton.jit
def mm_03(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        out_for_times: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b = tl.load(b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))
    
    for i in range(out_for_times):
        offs_am = (pid * out_for_times * BLOCK_SIZE_M + i * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask)
        accumulator = tl.dot(a, b)

        c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul_003(a, b, core, kernel_for_times, block_m):
    M, K = a.shape
    K, N = b.shape

    BLOCK_M = block_m
    BLOCK_N = N
    BLOCK_K = K
    grid = (core,)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    mm_03[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), kernel_for_times, BLOCK_M, BLOCK_N, BLOCK_K)

    return c

@triton.jit
def mm_04_1(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_am = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_bn = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = tl.load(b_ptrs)
    a = tl.load(a_ptrs)

    c = tl.dot(a, b)

    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def triton_matmul_004_1(a, b, block_m, block_n, block_k):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, block_n, block_k
    c = torch.empty((M, N), device="npu", dtype=torch.float16)
    grid = (triton.cdiv(N, BLOCK_N),)
    mm_04_1[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), BLOCK_M, BLOCK_N, BLOCK_K)

    return c

@triton.jit
def mm_04_2(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_am = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_bn = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(k + offs_k[None, :]) < K, other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None]) < K, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc.to(tl.float16)

    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul_004_2(a, b, block_m, block_n, block_k):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = block_m, block_n, block_k
    c = torch.empty((M, N), device="npu", dtype=torch.float16)
    grid = (triton.cdiv(N, BLOCK_N),)
    mm_04_2[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), BLOCK_M, BLOCK_N, BLOCK_K)
    return c

@impl(npu_lib, "triton_matmul")
def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    c = None
    st = strategy.get((M, N, K), None)
    if st is not None:
        op = st["op"]
        tiling = st['block']
        time = st['time']
        if op == "mm_01":
            block_m, block_n, block_k = tiling
            c = triton_matmul_001(a, b, block_m, block_n, block_k)
        elif op == "mm_03":
            core, kernel_for_times, block_m = tiling
            c = triton_matmul_003(a, b, core, kernel_for_times, block_m)
        elif op == "mm_04_1":
            block_m, block_n, block_k = tiling
            c = triton_matmul_004_1(a, b, block_m, block_n, block_k)
        elif op == "mm_04_2":
            block_m, block_n, block_k = tiling
            c = triton_matmul_004_2(a, b, block_m, block_n, block_k)
        else:
            c = torch.mm(a, b)
    else:
        c = torch.mm(a, b)
    return c

@impl(npu_meta, "triton_matmul")
def triton_matmul_fake(
    a: torch.Tensor,
    b: torch.Tensor,
):
    device = a.device
    dtype = a.dtype
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    return c
