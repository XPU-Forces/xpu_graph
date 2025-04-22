import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def fused_div_mul_sum_kernel(
        out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3,
        XBLOCK0: tl.constexpr, XBLOCK0_SUB: tl.constexpr,
        XBLOCK1: tl.constexpr, XBLOCK1_SUB: tl.constexpr,
        TSIZE0: tl.constexpr, XSIZE0: tl.constexpr, XSIZE1: tl.constexpr, YSIZE: tl.constexpr, RSIZE: tl.constexpr,
        CORE_SCALE: tl.constexpr):
    pid = tl.program_id(0)
    x0_pid = pid // CORE_SCALE
    x1_pid = pid % CORE_SCALE
    x0_offset = x0_pid * XBLOCK0
    x1_offset = x0_pid * XSIZE1 + x1_pid * XBLOCK1
    x0_sub_idx = tl.arange(0, XBLOCK0_SUB)
    x1_sub_idx = tl.arange(0, XBLOCK1_SUB)
    y_idx = tl.arange(0, YSIZE)
    x0_sub_loops = (XBLOCK0 + XBLOCK0_SUB - 1) // XBLOCK0_SUB
    x1_sub_loops = (XBLOCK1 + XBLOCK1_SUB - 1) // XBLOCK1_SUB
    r_idx = tl.arange(0, RSIZE)
    for x0_sub_id in tl.range(x0_sub_loops):
        x0_idx_offset = x0_offset + x0_sub_id * XBLOCK0_SUB
        x0_idx = x0_idx_offset + x0_sub_idx
        x0yr_offset = x0_idx[:, None, None] * YSIZE * RSIZE + y_idx[None, :, None] * RSIZE + r_idx[None, None, :]
        x0y_offset = x0_idx[:, None, None] * YSIZE + y_idx[None, :, None]
        x0yr_mask = x0_idx[:, None, None] < TSIZE0 * XSIZE0
        x0y_mask = x0yr_mask
        x0 = tl.load(in_ptr0 + x0yr_offset, mask = x0yr_mask, other = 0.0)
        x1 = tl.load(in_ptr1 + x0y_offset, mask = x0y_mask, other = 1.0)
        div_01 = x0 / x1
        for x1_sub_id in tl.range(x1_sub_loops):
            x1_idx = x1_offset + x1_sub_id * XBLOCK1_SUB + x1_sub_idx
            x1yr_offset = x1_idx[:, None, None] * YSIZE * RSIZE + y_idx[None, :, None] * RSIZE + r_idx[None, None, :]
            x1y_offset = x1_idx[:, None, None] * YSIZE + y_idx[None, :, None]
            x1yr_mask = x1_idx[:, None, None] < TSIZE0 * XSIZE1
            x1y_mask = x1yr_mask
            x2 = tl.load(in_ptr2 + x1yr_offset, mask = x1yr_mask, other = 0.0)
            x3 = tl.load(in_ptr3 + x1y_offset, mask = x1y_mask, other = 1.0)
            div_23 = x2 / x3
            div_01_4d = tl.reshape(div_01, [XBLOCK0_SUB, 1, YSIZE, RSIZE])
            div_23_4d = tl.reshape(div_23, [1, XBLOCK1_SUB, YSIZE, RSIZE])
            mul_0 = div_01_4d * div_23_4d
            sum_0 = tl.sum(mul_0, axis=3)
            sum_1 = sum_0 * 0.5
            x1rem_idx_offset = x1_offset % XSIZE1 + x1_sub_id * XBLOCK1_SUB
            x1rem_idx = x1rem_idx_offset + x1_sub_idx
            xx_offset = x0_idx[:, None, None] * XSIZE1 * YSIZE + x1rem_idx[None, :, None] * YSIZE + y_idx[None, None, :]
            xx_mask = (x0_idx[:, None, None] < TSIZE0 * XSIZE0) & (x1rem_idx[None, :, None] < XSIZE1)
            tl.store(out_ptr0 + xx_offset, sum_1, mask=xx_mask)

from torch.library import Library, impl
from xpu_graph.passes.patterns.targets.npu.triton_kernel import npu_def, npu_lib, npu_meta
npu_def.define("fused_div_mul_sum(Tensor div1_input, Tensor div1_divisor, Tensor div2_input, Tensor div2_divisor) -> (Tensor)")

@impl(npu_lib, "fused_div_mul_sum")
def fused_div_mul_sum(
    div1_input: torch.Tensor,
    div1_divisor: torch.Tensor,
    div2_input: torch.Tensor,
    div2_divisor: torch.Tensor,
) -> torch.Tensor:
    N0 = div1_input.shape[0]
    N1 = div1_input.shape[1]
    N2 = div2_input.shape[2]
    N3 = div1_input.shape[3]
    N4 = div1_input.shape[4]
    output_tensor = torch.empty((N0, N1, N2, N3), device=div1_input.device, dtype=div1_input.dtype)
    core_scale = 4
    fused_div_mul_sum_kernel[N0 * core_scale, 1, 1](
        output_tensor, div1_input, div1_divisor, div2_input, div2_divisor,
        XBLOCK0 = N1, XBLOCK0_SUB = N1,
        XBLOCK1 = (N2+core_scale-1)//core_scale, XBLOCK1_SUB = 2,
        TSIZE0 = N0, XSIZE0 = N1, XSIZE1 = N2, YSIZE = N3, RSIZE = N4,
        CORE_SCALE = core_scale)
    return output_tensor

@impl(npu_meta, "fused_div_mul_sum")
def fused_div_mul_sum_fake(
    div1_input: torch.Tensor,   
    div1_divisor: torch.Tensor,  
    div2_input: torch.Tensor,   
    div2_divisor: torch.Tensor,   
    ) -> torch.Tensor:
    shape1 = div1_input.shape
    shape2 = div2_input.shape
    ret_shape = (shape1[0], shape1[1], shape2[2], shape2[3])
    output_tensor = torch.empty(ret_shape, device=div1_input.device, dtype=div1_input.dtype)
    return output_tensor
