import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    is_similar,
    maybe_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    return outputs


def fn1(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) * torch.rsqrt(1e-6 + variance)
    outputs = bias + normalized
    return outputs


def fn2(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) / torch.sqrt(variance + 1e-5)
    return normalized


def fn3(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    return weight * normalized


def fn4(inputs, weight, bias):
    return fn0(inputs.to(torch.float32), weight, bias).to(inputs.dtype)


def fn5(inputs, weight, bias):
    return fn0(inputs.to(torch.float32), weight, bias).to(inputs.dtype)


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024,), device=device, dtype=data_type)
    bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0 or func == fn1:
        bias = torch.randn((1024,), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn2 or func == fn3:
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn4:
        inputs = torch.randn((8, 1024), device=device, dtype=torch.float32)
        weight = torch.randn((1024,), device=device, dtype=torch.float16)
        bias = torch.randn((1024,), device=device, dtype=torch.float16)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn5:
        inputs = torch.randn((8, 1024), device=device, dtype=torch.float16)
        weight = torch.randn((1024,), device=device, dtype=torch.float32)
        bias = torch.randn((1024,), device=device, dtype=torch.float32)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)


def layernorm_test_with_loss_and_grad(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type, requires_grad=True)
    weight = torch.randn((1024,), device=device, dtype=data_type, requires_grad=True)
    bias = torch.randn((1024,), device=device, dtype=data_type, requires_grad=True)
    dnorm = torch.randn((8, 1024), device=device, dtype=data_type)
    if func == fn4:
        weight = torch.randn((8, 1024), device=device, dtype=torch.float16, requires_grad=True)
        bias = torch.randn((8, 1024), device=device, dtype=torch.float16, requires_grad=True)
    if func == fn5:
        inputs = torch.randn((1024,), device=device, dtype=torch.float16, requires_grad=True)
        dnorm = torch.randn((1024,), device=device, dtype=torch.float16)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    norm0 = compiled(inputs, weight, bias)
    dinputs0, dweight0, dbias0 = torch.autograd.grad((norm0,), (inputs, weight, bias), (dnorm,), allow_unused=True)

    norm1 = func(inputs, weight, bias)
    dinputs1, dweight1, dbias1 = torch.autograd.grad((norm1,), (inputs, weight, bias), (dnorm,), allow_unused=True)

    assert is_similar(norm0, norm1)
    assert is_similar(dinputs0, dinputs1)
    assert maybe_similar(dweight0, dweight1)
    assert maybe_similar(dbias0, dbias1)


class TestLayerNorm:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4, fn5],
    )
    def test_layernorm_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            layernorm_test(self.infer_backend, pattern_func)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text
        if pattern_func in [fn5]:
            assert "Pattern.RemoveLayerNormCast" in caplog.text

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4, fn5],
    )
    def test_layernrom_patterns_with_loss_and_grad(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            layernorm_test_with_loss_and_grad(self.train_backend, pattern_func)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text
        if pattern_func in [fn5]:
            assert "Pattern.RemoveLayerNormCast" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    layernorm_test(infer_backend, fn0)
    layernorm_test(infer_backend, fn1)
    layernorm_test(infer_backend, fn2)
    layernorm_test(infer_backend, fn3)
    layernorm_test(infer_backend, fn4)
    layernorm_test(infer_backend, fn5)

    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    layernorm_test_with_loss_and_grad(train_backend, fn0)
    layernorm_test_with_loss_and_grad(train_backend, fn1)
    layernorm_test_with_loss_and_grad(train_backend, fn2)
    layernorm_test_with_loss_and_grad(train_backend, fn3)
    layernorm_test_with_loss_and_grad(train_backend, fn4)
    layernorm_test_with_loss_and_grad(train_backend, fn5)
