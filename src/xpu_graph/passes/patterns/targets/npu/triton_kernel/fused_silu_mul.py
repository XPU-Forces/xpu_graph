import logging

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry


@libentry()
@triton.jit
def fused_triton_mul_silu(
    output_ptr, x_ptr, BLOCK_SIZE: tl.constexpr, N: tl.constexpr  # 输出张量指针  # 输入张量指针  # 分块大小  # 输入张量最后一维的长度（D）
):
    pid = tl.program_id(0)  # 批次维度的 ID
    offset = pid * N  # 当前批次的起始位置

    for start in range(0, N // 2, BLOCK_SIZE):  # 遍历前半部分
        # 加载 x1（前半部分）
        x1_offsets = offset + start + tl.arange(0, BLOCK_SIZE)
        x1_mask = (start + tl.arange(0, BLOCK_SIZE)) < (N // 2)
        x1 = tl.load(x_ptr + x1_offsets, mask=x1_mask).to(tl.float32)
        # 加载 x2（后半部分）
        x2_offsets = offset + (N // 2) + start + tl.arange(0, BLOCK_SIZE)
        x2_mask = (start + tl.arange(0, BLOCK_SIZE)) < (N // 2)
        x2 = tl.load(x_ptr + x2_offsets, mask=x2_mask).to(tl.float32)
        # 计算 silu(x1) * x2
        result = x1 * tl.sigmoid(x1) * x2
        # 存储到 output_ptr（输出张量）
        out_offsets = pid * (N // 2) + start + tl.arange(0, BLOCK_SIZE)
        tl.store(output_ptr + out_offsets, result, mask=x1_mask)


from torch.library import Library, impl

from xpu_graph.passes.patterns.targets.npu.triton_kernel import (
    npu_def,
    npu_lib,
    npu_meta,
)

npu_def.define("fused_silu_mul(Tensor input) -> (Tensor)")


@impl(npu_lib, "fused_silu_mul")
def fused_silu_mul(
    input: torch.Tensor,
) -> torch.Tensor:
    shape = input.shape
    triton_res = torch.empty([shape[0], shape[1] // 2], dtype=input.dtype, device=input.device)
    core = shape[0]
    fused_triton_mul_silu[core, 1, 1](triton_res, input, BLOCK_SIZE=8192, N=shape[-1])
    return triton_res


@impl(npu_meta, "fused_silu_mul")
def fused_silu_mul(
    input: torch.Tensor,
) -> torch.Tensor:
    shape = input.shape
    triton_res = torch.empty([shape[0], shape[1] // 2], dtype=input.dtype, device=input.device)
    return triton_res
