from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx

from xpu_graph.utils import logger

from .check_ops import check_t_op, check_trans_op


def replace_with_module(gm: fx.GraphModule, node: fx.Node, module_name: str, params: tuple):
    with gm.graph.inserting_before(node):
        new_node = gm.graph.call_module(module_name, args=params)
    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)
    return new_node


class DefaultRMSNorm(torch.nn.Module):
    def forward(self, x, weight, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if weight is not None:
            weight = weight.to(torch.float32)
        return F.rms_norm(x, x.shape[-1:], weight=weight, eps=eps).to(input_dtype)


class DefaultLayerNorm(torch.nn.Module):
    def forward(self, x, weight, bias, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if weight is not None:
            weight = weight.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)
        return F.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps).to(input_dtype)


@torch.library.custom_op("xpu_graph::sdpa_wrapped_scale", mutates_args=())
def sdpa_wrapped_scale(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    scale = scale.item()
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)


@torch.library.register_fake("xpu_graph::sdpa_wrapped_scale")
def _(q, k, v, attn_mask, scale):
    scale = 1.0
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)


class SDPAWrappedScale(nn.Module):
    def forward(self, q, k, v, attn_mask, scale):
        return torch.ops.xpu_graph.sdpa_wrapped_scale(q, k, v, attn_mask, scale)


class DefaultSliceSumCatModule(torch.nn.Module):
    def __init__(self, slice_param):
        """
        Args:
            slice_param (list of tuples): A list of slice indices, where each tuple
                                          contains (start_idx, end_idx) for slicing.
        """
        super().__init__()

        slice_ = []
        for param in slice_param:
            slice_ += [param[0], param[1]]

        self.slice_param_list = slice_param

    def forward(self, input):
        """
        Forward pass for the SliceSumCatOperation.

        Args:
            input (torch.Tensor): The input tensor of shape (batch, row, col).

        Returns:
            torch.Tensor: The output tensor of shape (batch, len(slice_param) * col). The processed tensor after slice -> sum -> cat operations.
        """
        target_tensors = []
        for slice_arg in self.slice_param_list:
            slice_tensor = input[:, slice_arg[0] : slice_arg[1], :]
            sum_tensor = torch.sum(slice_tensor, dim=[1])
            target_tensors.append(sum_tensor)
        return torch.cat(target_tensors, axis=-1)
