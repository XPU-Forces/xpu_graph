from typing import List

import torch
import torch_mlu
import triton
import triton.language as tl

from . import libentry
from .get_mlu_devinfo import get_device_properties


@libentry.libentry()
@triton.jit
def mlu_triton_slice_low_kernel(
    input_ptr,
    output_ptr,
    start_indices_ptr,
    total_jobs,
    slice_len,
    input_row,
    input_stride,
    BLOCK_SIZE_R: tl.constexpr = 16,
    BLOCK_SIZE_C: tl.constexpr = 128,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    block_jobs_r = block_jobs
    if program_id == program_dim - 1:
        block_jobs_r = total_jobs - block_jobs * (program_dim - 1)

    loop = (input_row + BLOCK_SIZE_R - 1) // BLOCK_SIZE_R
    for l in range(loop):
        for block_idx in range(block_jobs_r):
            slice_idx = block_idx + block_jobs * program_id
            start_index = tl.load(start_indices_ptr + slice_idx)

            input_block_ptr = tl.make_block_ptr(
                base=input_ptr,
                shape=(input_row, slice_len),
                strides=(input_stride, 1),
                offsets=(l * BLOCK_SIZE_R, start_index),
                block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                order=(1, 0),
            )
            value = tl.load(input_block_ptr, boundary_check=(0,), padding_option=0)
            output_block_ptr = tl.make_block_ptr(
                base=output_ptr,
                shape=(input_row * total_jobs, slice_len),
                strides=(slice_len, 1),
                offsets=(slice_idx * input_row + l * BLOCK_SIZE_R, 0),
                block_shape=(BLOCK_SIZE_R, BLOCK_SIZE_C),
                order=(1, 0),
            )
            tl.store(output_block_ptr, value, boundary_check=(0,))


@torch.library.custom_op("torch_mlu_triton::fused_slice_low", mutates_args=())
def fused_slice_low(
    src_tensor: torch.Tensor,
    start_indices: torch.Tensor,
    slice_len: int,
    n_rows: int,
    input_stride: int,
) -> torch.Tensor:
    props = get_device_properties()
    block_size_r = n_rows
    block_size_c = slice_len
    size_of_dtype = 2
    if src_tensor.dtype == torch.float32:
        size_of_dtype = 4
    nram_limit = props.max_nram_size
    if block_size_r * block_size_c * size_of_dtype > nram_limit:
        block_size_r = nram_limit // size_of_dtype // block_size_c

    num_slices = len(start_indices)
    output_tensors = torch.empty(
        (num_slices * src_tensor.shape[0], slice_len),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    grid = (props.total_cores, 1, 1)
    mlu_triton_slice_low_kernel[grid](
        src_tensor,
        output_tensors,
        start_indices,
        num_slices,
        slice_len,
        n_rows,
        input_stride,
        block_size_r,
        block_size_c,
    )

    return output_tensors


@fused_slice_low.register_fake
def fused_slice_low_fake(src_tensor, start_indices, slice_len, n_rows, input_stride):
    output_tensors = torch.empty(
        (len(start_indices) * src_tensor.shape[0], slice_len),
        device=src_tensor.device,
        dtype=src_tensor.dtype,
    )
    return output_tensors
