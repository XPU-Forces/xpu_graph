from torch import Tensor
from torch.utils._pytree import tree_map


class DemoSubTensor(Tensor):
    def __init__(self, elem):
        super().__init__()
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                return t.elem
            else:
                return t

        def wrap(t):
            if isinstance(t, Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))


import logging

import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar


def demo_func(x):
    return x * x


def subtensor_test(xpu_graph, func, x):
    x = DemoSubTensor(x)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    expect = func(x)
    res = compiled(x)
    assert isinstance(res, DemoSubTensor)
    assert is_similar(expect, res)


def subtensor_tensor_with_grad(xpu_graph, func, x):
    x = DemoSubTensor(x)
    x.requires_grad_(True)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    grad = torch.rand_like(x)
    expect = func(x)
    res = compiled(x)
    assert isinstance(res, DemoSubTensor)
    assert is_similar(expect, res)
    (grad_expect,) = torch.autograd.grad(expect, x, grad)
    (grad_res,) = torch.autograd.grad(res, x, grad)
    assert isinstance(grad_res, DemoSubTensor)
    assert is_similar(grad_expect, grad_res)


class TestSubTensor:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    x = torch.rand(100)

    @pytest.mark.parametrize("func, x", [(demo_func, x)])
    def test_subtensor_case(self, func, x):
        subtensor_test(self.xpu_graph, func, x)


class TestSubTensorGrad:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=True)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    x = torch.rand(100)

    @pytest.mark.xfail(reason="FIXME: subclass tensor support for training not checked yet")
    @pytest.mark.parametrize("func, x", [(demo_func, x)])
    def test_subtensor_grad_case(self, func, x):
        subtensor_tensor_with_grad(self.xpu_graph, func, x)


if __name__ == "__main__":
    x = torch.rand(100)

    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    subtensor_test(xpu_graph, demo_func, x)

    config = xpu_graph.config.XpuGraphConfig(is_training=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    subtensor_tensor_with_grad(xpu_graph, demo_func, x)
