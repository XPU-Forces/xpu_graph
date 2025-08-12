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


def fn0(inputs, weight):
    square = torch.square(inputs)
    mean = torch.mean(square, dim=-1, keepdim=True)
    root = torch.sqrt(mean + 1e-6)
    outputs = inputs / root * weight
    return outputs


def fn1(inputs, weight):
    square = inputs**2
    mean = torch.mean(square, dim=-1, keepdim=True)
    iroot = torch.rsqrt(mean + 1e-6)
    outputs = weight * (inputs * iroot)
    return outputs


def fn2(inputs, weight):
    square = inputs * inputs
    mean = torch.mean(square, dim=-1, keepdim=True)
    root = (mean + 1e-3) ** 0.5
    normalized = inputs / root
    return normalized


def fn3(inputs, weight):
    square = inputs**2
    mean = torch.mean(square, dim=-1, keepdim=True)
    iroot = (1e-4 + mean) ** (-0.5)
    normalized = iroot * inputs
    return normalized


def fn4(inputs, weight):
    return fn0(inputs.to(torch.float32), weight).to(inputs.dtype)


def fn5(inputs, weight):
    return fn0(inputs.to(torch.float32), weight).to(inputs.dtype)


def rmsnorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0 or func == fn1:
        weight = torch.randn((1024), device=device, dtype=data_type)
        norm = compiled(inputs, weight)
        norm1 = func(inputs, weight)
        assert is_similar(norm1, norm)
    if func == fn2 or func == fn3:
        norm = compiled(inputs, weight)
        norm1 = func(inputs, weight)
        assert is_similar(norm1, norm)
    if func == fn4:
        inputs = torch.randn((8, 1024), device=device, dtype=torch.float32)
        weight = torch.randn((1024,), device=device, dtype=torch.float16)
        norm = compiled(inputs, weight)
        norm1 = func(inputs, weight)
        assert is_similar(norm1, norm)
    if func == fn5:
        inputs = torch.randn((8, 1024), device=device, dtype=torch.float16)
        weight = torch.randn((1024,), device=device, dtype=torch.float32)
        norm = compiled(inputs, weight)
        norm1 = func(inputs, weight)
        assert is_similar(norm1, norm)


def rmsnorm_test_with_loss_and_grad(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type, requires_grad=True)
    weight = torch.randn((1024,), device=device, dtype=data_type, requires_grad=True)
    dnorm = torch.randn((8, 1024), device=device, dtype=data_type)
    if func == fn4:
        weight = torch.randn((1024,), device=device, dtype=torch.float16, requires_grad=True)
    if func == fn5:
        inputs = torch.randn((1024,), device=device, dtype=torch.float16, requires_grad=True)
        dnorm = torch.randn((1024,), device=device, dtype=torch.float16)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    norm0 = compiled(inputs, weight)
    dinputs0, dweight0 = torch.autograd.grad((norm0,), (inputs, weight), (dnorm,), allow_unused=True)

    norm1 = func(inputs, weight)
    dinputs1, dweight1 = torch.autograd.grad((norm1,), (inputs, weight), (dnorm,), allow_unused=True)

    assert is_similar(norm0, norm1)
    assert is_similar(dinputs0, dinputs1)
    assert maybe_similar(dweight0, dweight1)


class TestRMSNorm:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4, fn5],
    )
    def test_rmsnorm_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            rmsnorm_test(self.infer_backend, pattern_func)
        assert "Pattern.FusedRMSNorm changed graph" in caplog.text
        if pattern_func in [fn5]:
            assert "Pattern.RemoveRMSNormCast" in caplog.text

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4, fn5],
    )
    def test_rmsnorm_patterns_with_loss_and_grad(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            rmsnorm_test_with_loss_and_grad(self.train_backend, pattern_func)
        assert "Pattern.FusedRMSNorm changed graph" in caplog.text
        if pattern_func in [fn5]:
            assert "Pattern.RemoveRMSNormCast" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    rmsnorm_test(infer_backend, fn0)
    rmsnorm_test(infer_backend, fn1)
    rmsnorm_test(infer_backend, fn2)
    rmsnorm_test(infer_backend, fn3)
    rmsnorm_test(infer_backend, fn4)
    rmsnorm_test(infer_backend, fn5)

    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    rmsnorm_test_with_loss_and_grad(train_backend, fn0)
    rmsnorm_test_with_loss_and_grad(train_backend, fn1)
    rmsnorm_test_with_loss_and_grad(train_backend, fn2)
    rmsnorm_test_with_loss_and_grad(train_backend, fn3)
    rmsnorm_test_with_loss_and_grad(train_backend, fn4)
    rmsnorm_test_with_loss_and_grad(train_backend, fn5)
