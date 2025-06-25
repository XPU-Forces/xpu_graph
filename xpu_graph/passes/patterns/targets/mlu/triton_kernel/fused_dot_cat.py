from typing import List

import torch
import torch_mlu
import triton
import triton.language as tl

from . import libentry
from .get_mlu_devinfo import get_device_properties


@triton.jit
def single_dot_cat(
    x,
    y,
    output,
    x_stride,
    y_stride,
    output_stride,
    x_row,
    y_row,
    output_row,
    slice_len,
    row_start_idx,
    output_start_offset,
    is_x_multi_dim: tl.constexpr,
    is_y_multi_dim: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_S1: tl.constexpr = 128,
    BLOCK_SIZE_S2: tl.constexpr = 128,
):
    if is_x_multi_dim:
        input_block_ptr0 = tl.make_block_ptr(
            base=x,
            shape=(x_row, slice_len),
            strides=(x_stride, 1),
            offsets=(row_start_idx, 0),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
            order=(1, 0),
        )
    else:
        input_block_ptr0 = tl.make_block_ptr(
            base=x,
            shape=(x_row, slice_len),
            strides=(x_stride, 1),
            offsets=(0, 0),
            block_shape=(1, BLOCK_SIZE_C),
            order=(1, 0),
        )
    if is_y_multi_dim:
        input_block_ptr1 = tl.make_block_ptr(
            base=y,
            shape=(y_row, slice_len),
            strides=(y_stride, 1),
            offsets=(row_start_idx, 0),
            block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
            order=(1, 0),
        )
    else:
        input_block_ptr1 = tl.make_block_ptr(
            base=y,
            shape=(y_row, slice_len),
            strides=(y_stride, 1),
            offsets=(0, 0),
            block_shape=(1, BLOCK_SIZE_C),
            order=(1, 0),
        )

    value0 = tl.load(input_block_ptr0, boundary_check=(0,), padding_option=0)
    value1 = tl.load(input_block_ptr1, boundary_check=(0,), padding_option=0)
    value0 = value0 * value1
    value0 = tl.reshape(value0, [BLOCK_SIZE_R, BLOCK_SIZE_S1, BLOCK_SIZE_S2])
    value = tl.empty([BLOCK_SIZE_R, BLOCK_SIZE_S2], dtype=value0.dtype)
    for i in range(BLOCK_SIZE_R):
        value[i : i + 1, :] = tl.sum(value0[i : i + 1, :, :], axis=1)
    output_block_ptr = tl.make_block_ptr(
        base=output,
        shape=(output_row, BLOCK_SIZE_S2 * 2),
        strides=(output_stride, 1),
        offsets=(row_start_idx, output_start_offset),
        block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_S2),
        order=(1, 0),
    )
    tl.store(output_block_ptr, value, boundary_check=(0,))


@libentry.libentry()
@libentry.libtuner(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_R": m,
            },
            num_stages=3,
            num_warps=1,
        )
        for m in [28, 28 * 2]
    ],
    key=[
        "is_input0_multi_dim",
        "is_input1_multi_dim",
        "is_input2_multi_dim",
        "is_input3_multi_dim",
        "BLOCK_SIZE_C",
        "BLOCK_SIZE_S1",
        "BLOCK_SIZE_S2",
    ],
)
@triton.jit
def mlu_triton_dot_cat_kernel(
    x0,
    y0,
    x1,
    y1,
    output,
    x0_stride,
    y0_stride,
    x1_stride,
    y1_stride,
    output_stride,
    total_jobs,
    x0_row,
    y0_row,
    x1_row,
    y1_row,
    slice_len,
    is_x0_multi_dim: tl.constexpr,
    is_y0_multi_dim: tl.constexpr,
    is_x1_multi_dim: tl.constexpr,
    is_y1_multi_dim: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 16,
    BLOCK_SIZE_S1: tl.constexpr = 128,
    BLOCK_SIZE_S2: tl.constexpr = 128,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    remainder = total_jobs % program_dim
    # by row(batch)
    if program_id < remainder:
        block_jobs_r = block_jobs + 1
        offset = program_id * (block_jobs + 1)
    else:
        block_jobs_r = block_jobs
        offset = remainder * (block_jobs + 1) + (program_id - remainder) * block_jobs

    loop = (block_jobs_r + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
    output_row = total_jobs

    for l in range(loop):
        row_start_idx = offset + l * BLOCK_SIZE_R
        single_dot_cat(
            x0,
            y0,
            output,
            x0_stride,
            y0_stride,
            output_stride,
            x0_row,
            y0_row,
            output_row,
            slice_len,
            row_start_idx,
            0,
            is_x0_multi_dim,
            is_y0_multi_dim,
            BLOCK_SIZE_R,
            BLOCK_SIZE_C,
            BLOCK_SIZE_S1,
            BLOCK_SIZE_S2,
        )
        single_dot_cat(
            x1,
            y1,
            output,
            x1_stride,
            y1_stride,
            output_stride,
            x1_row,
            y1_row,
            output_row,
            slice_len,
            row_start_idx,
            BLOCK_SIZE_S2,
            is_x1_multi_dim,
            is_y1_multi_dim,
            BLOCK_SIZE_R,
            BLOCK_SIZE_C,
            BLOCK_SIZE_S1,
            BLOCK_SIZE_S2,
        )


@torch.library.custom_op("torch_mlu_triton::fused_dot_cat", mutates_args=())
def fused_dot_cat_2inp(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    props = get_device_properties()
    input_row = max(x0.shape[0], y0.shape[0])
    _, s1, s2 = x0.shape
    output_tensors = torch.empty(
        (input_row, s2 * 2),
        device=x0.device,
        dtype=x0.dtype,
    )
    slice_len = s1 * s2

    grid = (props.total_cores, 1, 1)
    mlu_triton_dot_cat_kernel[grid](
        x0,
        y0,
        x1,
        y1,
        output_tensors,
        slice_len,
        slice_len,
        slice_len,
        slice_len,
        s2 * 2,
        input_row,
        x0.shape[0],
        y0.shape[0],
        x1.shape[0],
        y1.shape[0],
        slice_len,
        1 if x0.shape[0] > 1 else 0,
        1 if y0.shape[0] > 1 else 0,
        1 if x1.shape[0] > 1 else 0,
        1 if y1.shape[0] > 1 else 0,
        BLOCK_SIZE_C=s1 * s2,
        BLOCK_SIZE_S1=s1,
        BLOCK_SIZE_S2=s2,
    )

    return output_tensors


@fused_dot_cat_2inp.register_fake
def fused_dot_cat_2inp_fake(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    input_row = max(x0.shape[0], y0.shape[0])
    output_tensors = torch.zeros(
        (input_row, x0.shape[2] * 2),
        device=x0.device,
        dtype=x0.dtype,
    )
    return output_tensors
