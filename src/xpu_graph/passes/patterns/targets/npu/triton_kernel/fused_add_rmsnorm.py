import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime.libentry import libentry


@libentry()
@triton.jit
def fused_add_RMSNorm_kernel(
    # Pointers to x, residual, scale
    x_ptr,  # input
    r_ptr,  # residual
    s_ptr,  # weight(scale)
    rms_n_ptr,  # rms_norm
    new_r_prt,
    # dimensions
    d: tl.constexpr,
    # Meta-parameters
    XBLOCK_SUB: tl.constexpr,
):
    offset = tl.program_id(0) * d
    block = tl.arange(0, XBLOCK_SUB)
    mask = offset + block < offset + d
    x_ptrs = tl.load(x_ptr + block + offset, mask).to(tl.float32)
    r_ptrs = tl.load(r_ptr + block + offset, mask).to(tl.float32)
    s_ptrs = tl.load(s_ptr + block, mask).to(tl.float32)
    add = x_ptrs + r_ptrs
    new_r = add.to(tl.bfloat16)
    eps = 1e-6
    mean = tl.sum(add * add, -1) / d
    rms = tl.sqrt(mean + eps)
    x_hat = add / rms
    rms_norm = x_hat * s_ptrs
    tl.store(rms_n_ptr + block + offset, rms_norm, mask)
    tl.store(new_r_prt + block + offset, new_r, mask)


from torch.library import Library, impl

from xpu_graph.passes.patterns.targets.npu.triton_kernel import (
    npu_def,
    npu_lib,
    npu_meta,
)

npu_def.define("fused_add_rmsnorm(Tensor input, Tensor residual, Tensor weight) -> (Tensor, Tensor)")


@impl(npu_lib, "fused_add_rmsnorm")
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = input.shape
    rms_norm = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    residual_to = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    core = shape[0]
    fused_add_RMSNorm_kernel[core, 1, 1](input, residual, weight, rms_norm, residual_to, shape[1], shape[1])
    return rms_norm, residual_to


@impl(npu_meta, "fused_add_rmsnorm")
def fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rms_norm = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    residual_to = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    return rms_norm, residual_to
