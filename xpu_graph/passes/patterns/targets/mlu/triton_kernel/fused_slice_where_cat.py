import math

import torch
import torch_mlu
import triton
import triton.language as tl

from . import libentry
from .get_mlu_devinfo import get_device_properties


@libentry.libentry()
@triton.jit
def mlu_triton_slice_where_cat_kernel(
    output_ptr,
    where_ptr,
    slice_ptr,
    slice_param_ptr,
    output_stride,
    where_stride,
    slice_stride,
    slice_row,
    slice_len,
    slice_num,
    loop,
    total_jobs,
    BLOCK_SIZE_R: tl.constexpr = 32,
    BLOCK_SIZE_C: tl.constexpr = 32,
    BLOCK_SIZE_I: tl.constexpr = 32,
):
    program_id = tl.program_id(0)
    program_dim = tl.num_programs(axis=0)

    block_jobs = total_jobs // program_dim
    remainder_jobs = total_jobs % program_dim
    block_jobs_r = block_jobs + (1 if program_id < remainder_jobs else 0)
    block_start_index = program_id * block_jobs + min(program_id, remainder_jobs)

    for loop_ in range(loop):
        for block_idx in range(block_jobs_r):
            slice_idx = block_idx + block_start_index
            start_index = tl.load(slice_param_ptr + slice_idx)

            slice_block_ptr = tl.make_block_ptr(
                base=slice_ptr,
                shape=(slice_row, slice_len),
                strides=(slice_stride, 1),
                offsets=(loop_ * BLOCK_SIZE_R, start_index),
                block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                order=(1, 0),
            )
            slice_data = tl.load(slice_block_ptr, boundary_check=(0,), padding_option=0)

            where_block_ptr = tl.make_block_ptr(
                base=where_ptr,
                shape=(slice_row, 1),
                strides=(where_stride, 1),
                offsets=(loop_ * BLOCK_SIZE_R, 0),
                block_shape=(BLOCK_SIZE_R, 1),
                order=(1, 0),
            )
            where_data = tl.load(where_block_ptr, boundary_check=(0,), padding_option=0)

            slice_where = tl.where(where_data, 0, slice_data)

            output_block_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=(slice_row, slice_num * slice_len),
                strides=(output_stride, 1),
                offsets=(loop_ * BLOCK_SIZE_R, slice_idx * slice_len),
                block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                order=(1, 0),
            )
            tl.store(output_block_ptr, slice_where, boundary_check=(0,))


@torch.library.custom_op("torch_mlu_triton::fuse_slice_where_cat", mutates_args=(), device_types="mlu")
def fuse_slice_where_cat(
    where_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    slice_param: torch.Tensor,
    slice_len: int,
    slice_num: int,
) -> torch.Tensor:
    output_tensor = torch.empty(
        [slice_tensor.shape[0], slice_num * slice_len],
        dtype=slice_tensor.dtype,
        device=slice_tensor.device,
    )
    props = get_device_properties()
    where_stride = where_tensor.stride(0)
    slice_stride = slice_tensor.stride(0)
    output_stride = output_tensor.stride(0)

    slice_row = slice_tensor.shape[0]
    input_col = slice_tensor.shape[1]

    block_r = slice_row
    block_c = slice_len
    size_of_dtype = 2
    if slice_tensor.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = 416 * 1024
    # 2 -> 3: input(slice_data) + output(slice_where) + where_data + Caching of intermediate results
    if (block_r * block_c * size_of_dtype) * 3 > nram_limit:
        block_r = nram_limit // 3 // size_of_dtype // block_c

    loop = (slice_row + block_r - 1) // block_r

    mini_block = 4
    block_i = ((slice_num + mini_block - 1) // mini_block) * mini_block
    total_jobs = slice_num

    grid = (props.total_cores, 1, 1)
    mlu_triton_slice_where_cat_kernel[grid](
        output_tensor,
        where_tensor,
        slice_tensor,
        slice_param,
        output_stride,
        where_stride,
        slice_stride,
        slice_row,
        slice_len,
        slice_num,
        loop,
        total_jobs,
        block_r,
        block_c,
        block_i,
    )
    return output_tensor


@fuse_slice_where_cat.register_fake
def fuse_slice_where_cat_fake(
    where_tensor: torch.Tensor,
    slice_tensor: torch.Tensor,
    slice_param: torch.Tensor,
    slice_len: int,
    slice_num: int,
) -> torch.Tensor:
    output_tensor = torch.empty(
        [slice_tensor.shape[0], slice_num * slice_len],
        dtype=slice_tensor.dtype,
        device=slice_tensor.device,
    )
    return output_tensor
