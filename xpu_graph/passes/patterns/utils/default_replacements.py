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


class DenseParams:
    def __init__(self) -> None:
        self.input: Optional[fx.Node] = None
        self.weight: Optional[fx.Node] = None
        self.weight_trans: bool = False
        self.bias: Optional[Union[fx.Node, int, float]] = None
        self.act: str = "none"

    def as_tuple(self):
        return (
            self.input,
            self.weight,
            self.weight_trans,
            self.bias,
            self.act,
        )

    def set_params(self, input, weight, weight_trans, bias, act):
        self.input = input
        self.weight = weight
        self.weight_trans = weight_trans
        self.bias = bias
        self.act = act
        assert self.check_shape()
        return self

    def set_node(self, node):
        if len(node.args) != 5:
            return False
        self.input = node.args[0]
        self.weight = node.args[1]
        self.weight_trans = node.args[2]
        if not self.check_shape():
            return False

        if node.args[3] is not None:
            if not self.set_bias(node.args[3]):
                return False

        if not self.set_act(node.args[4]):
            return False

        return True

    def set_weight(self, node):
        if check_trans_op(node):
            trans_param = (node.args[1], node.args[2])
            if trans_param in [(0, 1), (1, 0), (-2, -1), (-1, -2)]:
                self.weight_trans = True
                node = node.args[0]
            else:
                return False
        elif check_t_op(node):
            self.weight_trans = True
            node = node.args[0]
        weight_shape = node.meta["val"].shape
        if len(weight_shape) != 2:
            logger.warning(f"MatMul pass: Unsupported weight dim {weight_shape}")
            return False
        self.weight = node

        if self.input:
            return self.check_shape()
        return True

    def set_input(self, node):
        input_shape = node.meta["val"].shape
        if len(input_shape) != 2:
            logger.warning(f"MatMul pass: Unsupported input dim {input_shape}")
            return False
        self.input = node

        if self.weight:
            return self.check_shape()
        return True

    def input_shape(self):
        return self.input.meta["val"].shape

    def weight_shape(self):
        weight_shape = self.weight.meta["val"].shape
        if self.weight_trans:
            return torch.Size([weight_shape[1], weight_shape[0]])
        return weight_shape

    def check_shape(self):
        if self.input is None:
            return False
        if self.weight is None:
            return False

        m1, k1 = self.input_shape()
        k2, n2 = self.weight_shape()
        if k1 != k2:
            logger.warning(
                f"MatMul pass: Unsupported dim input_shape: {self.input_shape()}, weight_shape: {self.weight_shape()}"
            )
            return False
        return True

    def set_bias(self, bias):
        if isinstance(bias, int):
            self.bias = bias
            return True
        if isinstance(bias, float):
            self.bias = bias
            return True

        m1, k1 = self.input_shape()
        k2, n2 = self.weight_shape()

        bias_shape = bias.meta["val"].shape
        try:
            shape_compat = torch.broadcast_shapes(torch.Size([m1, n2]), bias_shape) == torch.Size([m1, n2])
        except:
            shape_compat = False
        if not shape_compat:
            return False
        self.bias = bias
        return True

    def set_act(self, act_str):
        if act_str in ["gelu", "relu", "silu", "sigmoid", "none"]:
            self.act = act_str
            return True
        return False


class DefaultDenseLayer(nn.Module):
    def forward(self, input, weight, weight_trans, bias, act):
        if weight_trans:
            weight = weight.transpose(0, 1)
        if bias is None:
            y = torch.matmul(input, weight)
        elif not isinstance(bias, torch.Tensor):
            y = torch.matmul(input, weight) + bias
        else:
            y = torch.addmm(bias, input, weight)
        if act == "gelu":
            y = F.gelu(y)
        elif act == "relu":
            y = F.relu(y)
        elif act == "silu":
            y = F.silu(y)
        elif act == "sigmoid":
            y = F.sigmoid(y)
        return y


class BatchDenseParams:
    def __init__(self) -> None:
        self.input: Optional[fx.Node] = None
        self.weight: Optional[fx.Node] = None
        self.weight_trans: bool = False
        self.bias: Optional[fx.Node] = None
        self.act: str = "none"

    def as_tuple(self):
        return (
            self.input,
            self.weight,
            self.weight_trans,
            self.bias,
            self.act,
        )

    def set_node(self, node):
        if len(node.args) != 5:
            return False
        self.input = node.args[0]
        self.weight = node.args[1]
        self.weight_trans = node.args[2]
        if not self.check_shape():
            return False
        if node.args[3] is not None:
            if not self.set_bias(node.args[3]):
                return False
        if not self.set_act(node.args[4]):
            return False
        return True

    def set_params(self, input, weight, weight_trans, bias, act):
        self.input = input
        self.weight = weight
        self.weight_trans = weight_trans
        self.bias = bias
        self.act = act
        assert self.check_shape()
        return self

    def set_input(self, node):
        if len(node.meta["val"].shape) != 3:
            return False
        self.input = node
        if self.weight:
            return self.check_shape()
        return True

    def input_shape(self):
        return self.input.meta["val"].shape

    def set_weight(self, node):
        if len(node.meta["val"].shape) != 3:
            return False
        if check_trans_op(node):
            trans_param = (node.args[1], node.args[2])
            if trans_param in [(1, 2), (2, 1), (-2, -1), (-1, -2)]:
                self.weight_trans = True
                node = node.args[0]
            else:
                return False
        elif check_t_op(node):
            self.weight_trans = True
            node = node.args[0]

        self.weight = node
        if self.input:
            return self.check_shape()
        return True

    def weight_shape(self):
        weight_shape = self.weight.meta["val"].shape
        if self.weight_trans:
            return torch.Size(weight_shape[:-2] + (weight_shape[-1], weight_shape[-2]))
        return weight_shape

    def set_bias(self, bias):
        if not isinstance(bias, fx.Node):
            return False

        b1, m1, k1 = self.input_shape()
        b2, k2, n2 = self.weight_shape()

        bias_shape = bias.meta["val"].shape
        try:
            shape_compat = torch.broadcast_shapes(torch.Size([b1, m1, n2]), bias_shape) == torch.Size([b1, m1, n2])
        except:
            shape_compat = False
        if not shape_compat:
            return False
        self.bias = bias
        return True

    def set_act(self, act_str):
        if act_str in ["gelu", "silu", "none"]:
            self.act = act_str
            return True
        return False

    def check_shape(self):
        if self.input is None:
            return False
        if self.weight is None:
            return False
        b1, m1, k1 = self.input_shape()
        b2, k2, n2 = self.weight_shape()
        if (b1 != b2) or (k1 != k2):
            logger.warning(
                f"BatchMatmul pass: Unsupported dim input_shape: {self.input_shape()}, weight_shape: {self.weight_shape()}"
            )
            return False
        return True


class DefaultBatchDenseLayer(nn.Module):
    def forward(self, input, weight, weight_trans, bias, act_str):
        if weight_trans:
            weight = weight.transpose(-2, -1)

        if bias is not None:
            y = torch.baddbmm(bias, input, weight)
        else:
            y = torch.bmm(input, weight)

        if act_str == "gelu":
            y = F.gelu(y)
        elif act_str == "silu":
            y = F.silu(y)
        return y


class DefaultSDPA(nn.Module):
    def forward(self, q, k, v, attn_mask, scale):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale)
