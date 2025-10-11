import math
from typing import List

import torch
import torch_mlu
import triton
import triton.language as tl

from . import libentry
from .get_mlu_devinfo import get_device_properties

dtype_dict = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}


@libentry.libentry()
@triton.jit
def mlu_triton_slice_sum_cat_kernel(
    output_ptr,
    input_ptr,
    indices_ptr,
    input_stride0,
    input_stride1: tl.constexpr,
    output_stride0,
    total_jobs,
    row,
    col,
    output_num: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr = 32,
    BLOCK_SIZE_I: tl.constexpr = 32,
    BLOCK_SIZE_R: tl.constexpr = 32,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    block_jobs_r = block_jobs
    if program_id == program_dim - 1:
        block_jobs_r = total_jobs - block_jobs * (program_dim - 1)
    indices_offset = indices_ptr
    offset_param = tl.arange(0, BLOCK_SIZE_I)
    offset_row = tl.arange(0, BLOCK_SIZE_R)
    offset_col = tl.arange(0, BLOCK_SIZE_C)
    mask_indices = offset_param < output_num * 2
    indices = tl.load(indices_offset + offset_param, mask=mask_indices)

    mask_row = offset_row < row
    mask = mask_row[:, None]

    for block_idx in range(block_jobs_r):
        offset_id = block_idx + block_jobs * program_id
        input_offset = input_ptr + offset_id * input_stride0
        output_offset = output_ptr + offset_id * output_stride0

        for i in tl.static_range(output_num):
            # slice from input_data on chip
            indices_start = indices[i * 2]
            indices_end = indices[i * 2 + 1]
            valid_rows = (offset_row >= indices_start) & (offset_row < indices_end)
            slice_data = tl.load(
                input_offset + offset_row[:, None] * input_stride1 + offset_col[None, :],
                mask=valid_rows[:, None],
                other=0,
            )
            sum_value = tl.sum(slice_data, axis=0)
            tl.store(output_offset + i * col + offset_col, sum_value)


@libentry.libentry()
@triton.jit
def mlu_triton_slice_sum_cat_kernel_opt(
    output_ptr,
    input_ptr,
    indices_ptr,
    input_stride0,
    input_stride1: tl.constexpr,
    output_stride0,
    total_jobs,
    row,
    col,
    output_num: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr = 32,
    BLOCK_SIZE_I: tl.constexpr = 32,
    BLOCK_SIZE_R: tl.constexpr = 32,
    BLOCK_SIZE_C: tl.constexpr = 32,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    block_jobs_r = block_jobs
    if program_id == program_dim - 1:
        block_jobs_r = total_jobs - block_jobs * (program_dim - 1)
    indices_offset = indices_ptr
    offset_param = tl.arange(0, BLOCK_SIZE_I)
    offset_row = tl.arange(0, BLOCK_SIZE_R)
    offset_col = tl.arange(0, BLOCK_SIZE_C)
    offset_col_all_out = tl.arange(0, BLOCK_SIZE_C * output_num)
    mask_indices = offset_param < output_num * 2
    indices = tl.load(indices_offset + offset_param, mask=mask_indices)

    mask_row = offset_row < row
    mask = mask_row[:, None]

    for block_idx in range(block_jobs_r):
        offset_id = block_idx + block_jobs * program_id
        input_offset = input_ptr + offset_id * input_stride0
        output_offset = output_ptr + offset_id * output_stride0

        input_data = tl.load(
            input_offset + offset_row[:, None] * input_stride1 + offset_col[None, :],
            mask=mask,
        )

        sum_value = tl.empty([BLOCK_SIZE_C * output_num], dtype=input_data.dtype)
        for i in tl.static_range(output_num):
            # slice from input_data on chip
            indices_start = indices[i * 2]
            indices_end = indices[i * 2 + 1]
            valid_rows = (offset_row >= indices_start) & (offset_row < indices_end)
            combined_mask = valid_rows[:, None]
            slice_data = tl.where(combined_mask, input_data, 0)
            sum_value[i * BLOCK_SIZE_C : (i + 1) * BLOCK_SIZE_C] = tl.sum(slice_data, axis=0)

        tl.store(output_offset + offset_col_all_out, sum_value)


@torch.library.custom_op("torch_mlu_triton::fuse_slice_sum_cat", mutates_args=(), device_types="mlu")
def fuse_slice_sum_cat(
    input_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    output_num: int,
    end_row: int,
) -> torch.Tensor:
    props = get_device_properties()
    output_tensor = torch.empty(
        [input_tensor.shape[0], output_num * input_tensor.shape[2]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    input_stride0 = input_tensor.stride(0)
    input_stride1 = input_tensor.stride(1)
    batch = input_tensor.shape[0]
    col = input_tensor.shape[2]
    mini_block = 4
    block_size_t = 1
    block_size_i = ((output_num * 2 + mini_block - 1) // mini_block) * mini_block
    block_size_r = ((end_row + mini_block - 1) // mini_block) * mini_block
    block_size_c = col
    grid = (props.total_cores, 1, 1)

    mlu_triton_slice_sum_cat_kernel_opt[grid](
        output_tensor,
        input_tensor,
        slice_tensor,
        input_stride0,
        input_stride1,
        output_tensor.stride(0),
        batch,
        end_row,
        col,
        output_num,
        block_size_t,
        block_size_i,
        block_size_r,
        block_size_c,
        num_stages=3,
    )

    return output_tensor


@fuse_slice_sum_cat.register_fake
def fuse_slice_sum_cat_fake(
    input_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    output_num: int,
    end_idx: int,
) -> torch.Tensor:
    output_tensor = torch.empty(
        [input_tensor.shape[0], output_num * input_tensor.shape[2]],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    return output_tensor
