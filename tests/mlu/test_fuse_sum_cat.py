import pytest

import torch
import torch_mlu
import xpu_graph

from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar

device = "mlu:0"
aten = torch.ops.aten


def fn0(inputs):
    a, b, c = inputs
    a = torch.sum(a, dim=[1], keepdim=True)
    b = torch.sum(b, dim=[1], keepdim=True)
    c = torch.sum(c, dim=[1], keepdim=True)
    output = torch.cat([a, b, c], dim=1)
    return output


def fn1(inputs):
    a, b, c, d = inputs
    a = torch.sum(a, dim=[1], keepdim=True)
    b = torch.sum(b, dim=[1], keepdim=True)
    c = torch.sum(c, dim=[1], keepdim=True)
    output = torch.cat([a, b, c, d], dim=1)
    return output


def fn2(inputs):
    a, b, c, d = inputs
    a = torch.sum(a, dim=[1], keepdim=True)
    b = torch.sum(b, dim=[1], keepdim=True)
    c = torch.sum(c, dim=[1], keepdim=True)
    output = torch.cat([d, a, b, c, d], dim=1)
    return output


def fn3(inputs):
    a, b, c = inputs
    a = torch.sum(a, dim=[1])
    b = torch.sum(b, dim=[1])
    c = torch.sum(c, dim=[1])
    output = torch.cat([a, b, c], dim=-1)
    return output


def fn4(inputs):
    a, b, c = inputs
    a1 = torch.sum(a, dim=[1])
    b1 = torch.sum(b, dim=[1])
    c1 = torch.sum(c, dim=[1])
    output1 = torch.cat([a1, b1, c1], dim=-1)
    return output1


def fn5(inputs):
    a, b, c = inputs
    a1 = torch.sum(a, dim=[1])
    b1 = torch.sum(b, dim=[1])
    c1 = torch.sum(c, dim=[1])
    output1 = torch.stack([a1, b1, c1])
    return output1


def sumcat_test(xpu_graph, func):
    if func in [fn0, fn1, fn2, fn3, fn5]:
        a = torch.randn(128, 64).to(device=device)
        b = torch.randn(128, 32).to(device=device)
        c = torch.randn(128, 300).to(device=device)
        d = torch.randn(128, 1).to(device=device)
        if func in [fn1, fn2]:
            args = [a, b, c, d]
        else:
            args = [a, b, c]
    else:
        a = torch.randn(128, 64, 32).to(device=device)
        b = torch.randn(128, 32, 32).to(device=device)
        c = torch.randn(128, 16, 32).to(device=device)
        args = [a, b, c]
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res1 = func(args)
    res = compiled(args)
    assert is_similar(res1.float(), res.float())


class TestSumCat:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig()
        config.target = xpu_graph.config.Target.mlu
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4, fn5],
    )
    def test_sumcat_patterns(self, pattern_func):
        sumcat_test(self.xpu_graph, pattern_func)


if __name__ == "__main__":
    pytest.main()
