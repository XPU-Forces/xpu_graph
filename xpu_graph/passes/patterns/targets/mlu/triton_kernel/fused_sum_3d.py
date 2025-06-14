from typing import List, Tuple

import torch
import torch_mlu
import triton
import triton.language as tl

from . import libentry
from .get_mlu_devinfo import get_device_properties


@triton.jit
def mlu_triton_sum_3d_input_dim_1_kernel(
    input_ptr,
    output_ptr,
    s0: tl.constexpr,
    s1: tl.constexpr,
    s2: tl.constexpr,
    BLOCK_SIZE_0: tl.constexpr = 16,
    BLOCK_SIZE_1: tl.constexpr = 16,
    BLOCK_SIZE_2: tl.constexpr = 16,
):
    loop = (s0 + BLOCK_SIZE_0 - 1) // BLOCK_SIZE_0
    for l in range(loop):
        offset_0 = l * BLOCK_SIZE_0
        input_ptr_ = tl.make_block_ptr(
            base=input_ptr,
            shape=(s0, s1, s2),
            strides=(s1 * s2, s2, 1),
            offsets=(offset_0, 0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2),
            order=(2, 1, 0),
        )
        data = tl.load(input_ptr_, boundary_check=(2, 1, 0))
        data = tl.sum(data, axis=1)

        output_ptr_ = tl.make_block_ptr(
            base=output_ptr,
            shape=(s0, s2),
            strides=(s2, 1),
            offsets=(offset_0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_2),
            order=(1, 0),
        )
        tl.store(output_ptr_, data, boundary_check=(1, 0))


@triton.jit
def mlu_triton_sum_3d_input_dim_2_kernel(
    input_ptr,
    output_ptr,
    s0,
    s1,
    s2,
    BLOCK_SIZE_0: tl.constexpr = 16,
    BLOCK_SIZE_1: tl.constexpr = 16,
    BLOCK_SIZE_2: tl.constexpr = 16,
):
    loop = (s0 + BLOCK_SIZE_0 - 1) // BLOCK_SIZE_0
    for l in range(loop):
        offset_0 = l * BLOCK_SIZE_0
        input_ptr_ = tl.make_block_ptr(
            base=input_ptr,
            shape=(s0, s1, s2),
            strides=(s1 * s2, s2, 1),
            offsets=(offset_0, 0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1, BLOCK_SIZE_2),
            order=(2, 1, 0),
        )
        data = tl.load(input_ptr_, boundary_check=(2, 1, 0))
        data = tl.sum(data, axis=2)

        output_ptr_ = tl.make_block_ptr(
            base=output_ptr,
            shape=(s0, s1),
            strides=(s1, 1),
            offsets=(offset_0, 0),
            block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1),
            order=(1, 0),
        )
        tl.store(output_ptr_, data, boundary_check=(1, 0))


@libentry.libentry()
@triton.jit
def mlu_triton_sum_3d_input_kernel(
    input_ptrs,
    output_ptrs,
    tuple_s0,
    tuple_s1,
    tuple_s2,
    dim: tl.constexpr,
    MUTI_BLOCK_SIZE_0,
    MUTI_BLOCK_SIZE_1,
    MUTI_BLOCK_SIZE_2,
    TOTAL_JOBS: tl.constexpr = 16,
):
    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    for sum_idx in tl.static_range(TOTAL_JOBS):
        if sum_idx % program_dim == program_id:
            s0 = tuple_s0[sum_idx]
            s1 = tuple_s1[sum_idx]
            s2 = tuple_s2[sum_idx]
            input_ptr = input_ptrs[sum_idx]
            output_ptr = output_ptrs[sum_idx]
            if dim == 1:
                mlu_triton_sum_3d_input_dim_1_kernel(
                    input_ptr,
                    output_ptr,
                    s0,
                    s1,
                    s2,
                    MUTI_BLOCK_SIZE_0[sum_idx],
                    MUTI_BLOCK_SIZE_1[sum_idx],
                    MUTI_BLOCK_SIZE_2[sum_idx],
                )
            else:
                mlu_triton_sum_3d_input_dim_2_kernel(
                    input_ptr,
                    output_ptr,
                    s0,
                    s1,
                    s2,
                    MUTI_BLOCK_SIZE_0[sum_idx],
                    MUTI_BLOCK_SIZE_1[sum_idx],
                    MUTI_BLOCK_SIZE_2[sum_idx],
                )


@torch.library.custom_op("torch_mlu_triton::fused_sum_3d_input", mutates_args=())
def fused_sum_3d_input(
    inputs: List[torch.Tensor],
    dim: int,
) -> List[torch.Tensor]:
    props = get_device_properties()
    outputs = []
    muti_block_s0 = []
    muti_block_s1 = []
    muti_block_s2 = []
    num_stages = 1
    for input in inputs:
        s0, s1, s2 = input.shape
        muti_block_s1.append(s1)
        muti_block_s2.append(s2)
        max_s0 = min(props.max_nram_size // (s1 * s2 * 8 + 1), s0)
        # if max_s0 >= s0:
        #     num_stages = 1
        # else:
        #     max_s0 = max_s0 // 2
        #     num_stages = 3
        muti_block_s0.append(max_s0)

        shape = ()
        if dim == 0:
            shape = (s1, s2)
        elif dim == 1:
            shape = (s0, s2)
        else:
            shape = (s0, s1)
        output = torch.empty(
            shape,
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )
        outputs.append(output)
    # input must be contiguous
    mlu_triton_sum_3d_input_kernel[(props.total_cores,)](
        tuple([i.contiguous() for i in inputs]),
        tuple(outputs),
        tuple([tl.constexpr(i.shape[0]) for i in inputs]),
        tuple([tl.constexpr(i.shape[1]) for i in inputs]),
        tuple([tl.constexpr(i.shape[2]) for i in inputs]),
        dim,
        tuple([tl.constexpr(max_s0) for max_s0 in muti_block_s0]),
        tuple([tl.constexpr(max_s1) for max_s1 in muti_block_s1]),
        tuple([tl.constexpr(max_s2) for max_s2 in muti_block_s2]),
        len(inputs),
        num_stages=num_stages,
    )
    return outputs


@fused_sum_3d_input.register_fake
def fused_sum_3d_input_fake(
    inputs: List[torch.Tensor],
    dim: int,
) -> List[torch.Tensor]:
    outputs = []
    for input in inputs:
        s0, s1, s2 = input.shape
        shape = ()
        if dim == 0:
            shape = (s1, s2)
        elif dim == 1:
            shape = (s0, s2)
        else:
            shape = (s0, s1)
        output = torch.empty(
            shape,
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )
        outputs.append(output)
    return outputs
