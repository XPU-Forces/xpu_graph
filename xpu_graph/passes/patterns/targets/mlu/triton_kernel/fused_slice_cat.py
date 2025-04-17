import torch
import torch_mlu
import triton
import triton.language as tl
from . import libentry
from .get_mlu_devinfo import get_device_properties


@libentry.libentry()
@triton.jit
def mlu_triton_slice_cat_kernel(
    data_ptr,
    output_ptr,
    indices_ptr,
    stride,
    n_elements,
    total_jobs,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(1)
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    indices = tl.load(indices_ptr + offset, mask=mask, other=0)

    program_dim = tl.num_programs(axis=0)
    program_id = tl.program_id(0)
    block_jobs = total_jobs // program_dim
    block_jobs_r = block_jobs
    if program_id == program_dim - 1:
        block_jobs_r = total_jobs - block_jobs * (program_dim - 1)

    for i in range(block_jobs_r):
        row_id = block_jobs * program_id + i
        data_offset = row_id * stride + indices
        data = tl.load(data_ptr + data_offset, mask=mask, other=0)

        output_offset = row_id * n_elements + offset
        tl.store(output_ptr + output_offset, data, mask=mask)


@torch.library.custom_op("torch_mlu_triton::fused_slice_cat", mutates_args=())
def fused_slice_cat(
    input_tensor: torch.Tensor,
    indices_tensor: torch.Tensor,
    n_rows: int,
    elements: int,
    input_stride: int,
) -> torch.Tensor:
    props = get_device_properties()
    size_of_dtype = 2
    if input_tensor.dtype == torch.float32:
        size_of_dtype = 4
    # 4 is int32(indices)
    block_size = min(props.max_nram_size // (size_of_dtype + 4), elements)
    num_blocks = (elements + block_size - 1) // block_size
    output_tensor = torch.empty(
        (n_rows, elements), dtype=input_tensor.dtype, device=input_tensor.device
    )
    mlu_triton_slice_cat_kernel[(props.total_cores, num_blocks)](
        input_tensor,
        output_tensor,
        indices_tensor,
        input_stride,
        elements,
        n_rows,
        BLOCK_SIZE=block_size,
    )
    return output_tensor


@fused_slice_cat.register_fake
def fused_slice_cat_fake(input_tensor, indices_tensor, n_rows, elements, input_stride):
    return torch.empty(n_rows, elements, device=input_tensor.device, dtype=input_tensor.dtype)
