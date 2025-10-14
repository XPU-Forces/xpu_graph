import math
from typing import List

import torch
import torch_mlu
import triton
import triton.language as tl

dtype_dict1 = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
}


@triton.jit
def mlu_triton_sum_cat_3d_kernel(
    inputs_ptr,
    output_ptr,
    m_list_ptr,
    B,
    N: tl.constexpr,
    INPUT_NUM: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    batchs_per_job = tl.cdiv(B, num_jobs)
    b_start = pid * batchs_per_job

    if DTYPE == 3:
        dtype = tl.float32
    elif DTYPE == 2:
        dtype = tl.float16
    elif DTYPE == 1:
        dtype = tl.bfloat16

    m_list_offsets = tl.arange(0, INPUT_NUM)
    m_list = tl.load(m_list_ptr + m_list_offsets)
    input_ptrs = tl.load(inputs_ptr + m_list_offsets)

    output = tl.empty([BLOCK_BATCH, INPUT_NUM, N], dtype=dtype)
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, N)
    for batch_start in tl.range(0, batchs_per_job, BLOCK_BATCH, num_stages=2):
        b_offsets = b_start + batch_start + tl.arange(0, BLOCK_BATCH)
        for i in tl.range(0, INPUT_NUM, num_stages=2):
            offsets = b_offsets[:, None, None] * N * m_list[i] + m_offsets[None, :, None] * N + n_offsets[None, None, :]
            mask = (b_offsets < B)[:, None, None] & (m_offsets < m_list[i])[None, :, None]
            input = tl.load(input_ptrs[i].to(tl.pointer_type(dtype)) + offsets, mask=mask, other=0)
            n = min(BLOCK_BATCH, B - batch_start)
            for b in tl.range(0, n, num_stages=1):
                result = tl.sum(input[b, :, :], 0)
                output[b, i, :] = result
        output_offsets = (
            b_offsets[:, None, None] * N * INPUT_NUM
            + tl.arange(0, INPUT_NUM)[None, :, None] * N
            + tl.arange(0, N)[None, None, :]
        )
        mask = (b_offsets < B)[:, None, None]
        tl.store(output_ptr + output_offsets, output)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_BATCH": 1}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_BATCH": 4}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_BATCH": 16}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_BATCH": 64}, num_stages=3, num_warps=1),
        triton.Config(kwargs={"BLOCK_BATCH": 128}, num_stages=3, num_warps=1),
    ],
    key=["B", "INPUT_NUM"],
)
@triton.jit
def mlu_triton_sum_cat_2d_kernel(
    inputs_ptr,
    output_ptr,
    m_list_ptr,
    B,
    INPUT_NUM: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)

    if DTYPE == 3:
        dtype = tl.float32
    elif DTYPE == 2:
        dtype = tl.float16
    elif DTYPE == 1:
        dtype = tl.bfloat16

    batch_start = pid * BLOCK_BATCH * num_jobs
    step = BLOCK_BATCH

    m_list_offsets = tl.arange(0, INPUT_NUM)
    m_list = tl.load(m_list_ptr + m_list_offsets)
    input_ptrs = tl.load(inputs_ptr + m_list_offsets)
    for batch_idx in range(batch_start, B, step):
        batch_offsets = batch_idx + tl.arange(0, BLOCK_BATCH)
        output = tl.empty([INPUT_NUM, BLOCK_BATCH], dtype=dtype)
        for i in range(0, INPUT_NUM):
            m_offsets = tl.arange(0, BLOCK_SIZE)
            offsets = batch_offsets[:, None] * m_list[i] + m_offsets[None, :]
            mask = (batch_offsets < B)[:, None] & (m_offsets < m_list[i])[None, :]
            input = tl.load(
                input_ptrs[i].to(tl.pointer_type(dtype)) + offsets,
                mask=mask,
                other=0,
            )
            result = tl.sum(input, 1)
            output[i, :] = result
        output_offsets = tl.arange(0, INPUT_NUM)[:, None] * B + batch_offsets[None, :]
        mask = (batch_offsets < B)[None, :]
        tl.store(output_ptr + output_offsets, output, mask=mask)


@torch.library.custom_op("torch_mlu_triton::fuse_sum_cat_2d", mutates_args={}, device_types="mlu")
def fuse_sum_cat_2d(
    inputs: List[torch.Tensor],
    lengths_tensor: torch.Tensor,
    dim_0: int,
    block_size: int = 64,
) -> torch.Tensor:
    device = inputs[0].device
    dtype = inputs[0].dtype
    input_tensor = torch.empty(
        (len(inputs)),
        device=device,
        dtype=torch.long,
    )
    for i in range(len(inputs)):
        input_tensor[i].fill_(inputs[i].data_ptr())

    output_tensor = torch.empty(
        (len(inputs), dim_0),
        device=device,
        dtype=dtype,
    )

    grid = lambda meta: (
        min(
            triton.cdiv(dim_0, meta["BLOCK_BATCH"]),
            torch.mlu.get_device_properties(0).multi_processor_count,
        ),
    )
    mlu_triton_sum_cat_2d_kernel[grid](
        input_tensor,
        output_tensor,
        lengths_tensor,
        dim_0,
        len(inputs),
        BLOCK_SIZE=block_size,
        DTYPE=dtype_dict1[dtype],
    )
    return output_tensor


@fuse_sum_cat_2d.register_fake
def fuse_sum_cat_2d_fake(
    inputs: List[torch.Tensor],
    lengths_tensor: torch.Tensor,
    dim_0: int,
    block_size: int,
):
    device = inputs[0].device
    dtype = inputs[0].dtype
    return torch.empty(len(inputs), dim_0, device=device, dtype=dtype)


@torch.library.custom_op("torch_mlu_triton::fuse_sum_cat_3d", mutates_args=(), device_types="mlu")
def fuse_sum_cat_3d(
    inputs: List[torch.Tensor],
    dim_0: int,
    dim_1_tensor: torch.Tensor,
    dim_2: int,
    num_tensors: int,
    max_sequence_length: int,
) -> torch.Tensor:
    device = inputs[0].device
    dtype = inputs[0].dtype
    output_tensor = torch.empty(dim_0, dim_2 * num_tensors, device=device, dtype=dtype)
    tensor_ptrs = torch.empty(
        (len(inputs)),
        device=device,
        dtype=torch.long,
    )
    for i in range(len(inputs)):
        tensor_ptrs[i].fill_(inputs[i].data_ptr())

    BLOCK_BATCH = 3
    grid_1 = min(
        math.ceil(dim_0 / BLOCK_BATCH),
        torch.mlu.get_device_properties(0).multi_processor_count,
    )
    grid = (grid_1, 1, 1)
    mlu_triton_sum_cat_3d_kernel[grid](
        tensor_ptrs,
        output_tensor,
        dim_1_tensor,
        dim_0,
        dim_2,
        num_tensors,
        BLOCK_BATCH,
        max_sequence_length,
        DTYPE=dtype_dict1[dtype],
    )
    return output_tensor


@fuse_sum_cat_3d.register_fake
def fuse_sum_cat_3d_fake(
    inputs: List[torch.Tensor],
    dim_0: int,
    dim_1: torch.Tensor,
    dim_2: int,
    num_tensors: int,
    max_sequence_length: int,
):
    device = inputs[0].device
    dtype = inputs[0].dtype
    output_tensor = torch.empty(dim_0, dim_2 * num_tensors, device=device, dtype=dtype)
    return output_tensor
