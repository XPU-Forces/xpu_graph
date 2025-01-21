import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu
from xpu_graph.config import OptLevel
import torch_mlu_ops
import xpu_graph
from xpu_graph.test_utils import is_similar
import pytest


device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(
        inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5
    )
    return output


def fn1(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(
        inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5
    )
    return output, inputs_


def fn2(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(
        inputs, dim=-1, keepdim=True, unbiased=False
    )  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    return outputs


def fn3(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(
        inputs, dim=-1, keepdim=True, unbiased=False
    )  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) * torch.rsqrt(1e-6 + variance)
    outputs = bias + normalized
    return outputs


def fn4(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) / torch.sqrt(variance + 1e-5)
    return normalized


def fn5(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    return weight * normalized


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    residual = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024), device=device, dtype=data_type)
    bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0:
        norm = compiled(inputs, residual, weight, bias)
        norm1 = func(inputs, residual, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn1:
        norm, res = compiled(inputs, residual, weight, bias)
        norm1, res1 = func(inputs, residual, weight, bias)
        assert is_similar(norm1, norm)
        assert is_similar(res1, res)
    if func == fn2 or func == fn3:
        bias = torch.randn((1024,), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn4 or func == fn5:
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn2],
    )
    def test_layernrom_patterns(self, pattern_func):
        layernorm_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)
    layernorm_test(xpu_graph_backend, fn0)
    layernorm_test(xpu_graph_backend, fn1)
    layernorm_test(xpu_graph_backend, fn2)
    layernorm_test(xpu_graph_backend, fn3)
    layernorm_test(xpu_graph_backend, fn4)
    layernorm_test(xpu_graph_backend, fn5)
