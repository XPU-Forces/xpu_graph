from typing import List

import torch
import torch_mlu


@torch.library.custom_op("torch_mlu_triton::fused_grouped_gemm", mutates_args=())
def fused_grouped_gemm(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    c: List[torch.Tensor],
    bias: List[torch.Tensor],
    alpha: List[float],
    beta: List[float],
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    output = torch.empty(len(a), a[0].shape[0], b[0].shape[1], device=a[0].device, dtype=a[0].dtype)
    output_list = [output[i] for i in range(len(a))]

    torch.ops.torch_mlu.grouped_gemm(a, b, c, bias, alpha, beta, trans_a, trans_b, out=output_list)
    return output


@fused_grouped_gemm.register_fake
def fused_grouped_gemm_fake(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    c: List[torch.Tensor],
    bias: List[torch.Tensor],
    alpha: List[float],
    beta: List[float],
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    output = torch.empty(len(a), a[0].shape[0], b[0].shape[1], device=a[0].device, dtype=a[0].dtype)
    return output
