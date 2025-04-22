import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.math as tl_math

@triton.jit
def fused_brc_permute_sum_kernel(
        out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, X4: tl.constexpr,
        XBLOCK0: tl.constexpr, XBLOCK0_SUB: tl.constexpr,
        XBLOCK1: tl.constexpr, XBLOCK1_SUB: tl.constexpr,
        PBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
        XSIZE0: tl.constexpr, XSIZE1: tl.constexpr,
        PSIZE: tl.constexpr, RSIZE: tl.constexpr,
        CORE_SCALE: tl.constexpr):
    tl.static_assert(XBLOCK0 == 1)
    tl.static_assert(XBLOCK0_SUB == 1)
    pid = tl.program_id(0)
    x0_pid = pid // CORE_SCALE
    x1_pid = pid
    x0_max = XSIZE0 * 1
    x1_max = XSIZE0 * XSIZE1
    p_sub_idx = tl.arange(0, PBLOCK)
    r_idx = tl.arange(0, RBLOCK)
    x0_sub_idx = tl.arange(0, XBLOCK0_SUB)
    x1_sub_idx = tl.arange(0, XBLOCK1_SUB)
    x_sub_loops = (XBLOCK1 + XBLOCK1_SUB - 1) // XBLOCK1_SUB
    p_sub_loops = (PSIZE + PBLOCK - 1) // PBLOCK
    for x_sub_id in tl.range(x_sub_loops):
        x0_idx = x0_pid * XBLOCK0 + 0 * XBLOCK0_SUB + x0_sub_idx
        x0pr_idx = x0_idx[:, None, None] * RSIZE + r_idx[None, None, :]
        x0pr_mask = (x0_idx[:, None, None] < x0_max) & (r_idx[None, None, :] < RSIZE)
        x3 = tl.load(in_ptr3 + x0pr_idx, mask=x0pr_mask, other=0.0)
        for p_sub_id in tl.range(p_sub_loops):
            x1_idx = x1_pid * XBLOCK1 + x_sub_id * XBLOCK1_SUB + x1_sub_idx
            p_idx = p_sub_id * PBLOCK + p_sub_idx
            x1pr_idx = x1_idx[:, None, None] * PSIZE * RSIZE + p_idx[None, :, None] * RSIZE + r_idx[None, None, :]
            x1pr_mask = (x1_idx[:, None, None] < x1_max) & ((p_idx[None, :, None] < PSIZE) & (r_idx[None, None, :] < RSIZE))
            x1rp_idx = x1_idx[:, None, None] * RSIZE * PSIZE + r_idx[None, :, None] * PSIZE + p_idx[None, None, :]
            x1rp_mask = (x1_idx[:, None, None] < x1_max) & ((r_idx[None, :, None] < RSIZE) & (p_idx[None, None, :] < PSIZE))
            x2 = tl.load(in_ptr2 + x1rp_idx, mask=x1rp_mask, other=0.0)
            tmp0 = tl.broadcast_to(x3, (XBLOCK0_SUB, PBLOCK, RBLOCK))
            tmp1 = tmp0.to(tl.float16)
            tmp2 = 1.0 - tmp1
            tmp3 = tmp2 != 0
            tmp4 = tl.where(tmp3, X4, tmp2)
            tmp5 = tl.permute(x2, (0, 2, 1))
            x1 = tl.load(in_ptr1 + x1pr_idx, mask=x1pr_mask, other=0.0)
            tmp6 = x1 + tmp5
            tmp7 = tmp6 * 0.07216878364870322
            x0 = tl.load(in_ptr0 + x1pr_idx, mask=x1pr_mask, other=0.0)
            tmp8 = x0 + tmp7
            tmp9 = tl.broadcast_to(tmp4, (XBLOCK0_SUB, PBLOCK, RBLOCK))
            tmp10 = tmp8 + tmp9
            tmp11 = tmp10.to(tl.float32)
            tmp12 = tl.max(tmp11, axis=2, keep_dims=True)
            tmp13 = tmp11 - tmp12
            tmp14 = tl_math.exp(tmp13)
            tmp15 = tl.sum(tmp14, axis=2, keep_dims=True)
            tmp16 = tmp14 / tmp15
            tmp17 = tmp16.to(tl.float16)
            o0r_idx = x1pr_idx
            o0r_mask = x1pr_mask
            tl.store(out_ptr0 + o0r_idx, tmp17, mask=o0r_mask)

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta
npu_def.define("fused_brc_permute_sum(Tensor view_7, Tensor buf47, Tensor buf59, Tensor arg107_1, float buf61) -> (Tensor)")

@impl(npu_lib, "fused_brc_permute_sum")
def fused_brc_permute_sum(
    view_7: torch.Tensor,
    buf47: torch.Tensor,
    buf59: torch.Tensor,
    arg107_1: torch.Tensor,
    buf61_val: float
) -> torch.Tensor:
    N0 = view_7.shape[0]
    N1 = view_7.shape[1]
    N2 = view_7.shape[2]
    N3 = view_7.shape[3]
    CORE_SCALE = 4
    out = torch.empty((N0, N1, N2, N3), dtype=view_7.dtype, device=view_7.device)
    fused_brc_permute_sum_kernel[N0 * CORE_SCALE, 1, 1](
        out, view_7, buf47, buf59, arg107_1,
        X4 = buf61_val,
        XBLOCK0 = 1, XBLOCK0_SUB = 1,
        XBLOCK1 = N1 // CORE_SCALE, XBLOCK1_SUB = 3,
        PBLOCK = 16, RBLOCK = N3,
        XSIZE0 = N0, XSIZE1 = N1,
        PSIZE = N2, RSIZE = N3,
        CORE_SCALE = CORE_SCALE)
    return out

@impl(npu_meta, "fused_brc_permute_sum")
def fused_brc_permute_sum_fake(
    view_7: torch.Tensor,
    buf47: torch.Tensor,
    buf59: torch.Tensor,
    arg107_1: torch.Tensor,
    buf61: float
) -> torch.Tensor:
    out = torch.empty(view_7.shape, device=view_7.device, dtype=view_7.dtype)
    return out
