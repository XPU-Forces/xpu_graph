import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def fn0(inputs):
    a, b, c = inputs
    a = torch.cat([a, b], dim=1)
    output = torch.cat([a, c], dim=1)
    return output


def cat_cat_test(xpu_graph, func, dynamic):
    a = torch.randn(128, 64)
    b = torch.randn(128, 32)
    c = torch.randn(128, 300)
    args = [a, b, c]

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    expect = func(args)
    res = compiled(args)
    assert is_similar(expect, res)


def fn1(input):
    return torch.cat([input], dim=1)


def fn2(a):
    outputs = a.split([2, 5, a.shape[-1] - 7], dim=-1)
    output = torch.cat(outputs, dim=-1)
    return output


def fn2_xfail(a):
    outputs = a.split([5, 5], dim=0)
    output = torch.cat(outputs, dim=-1)
    return output


def fn2_xfail2(a):
    outputs = a.split([2, 5, 3])
    output = torch.cat(outputs[:2])
    return output


def cat_test(xpu_graph, func, dynamic):
    input = torch.randn(10, 10)

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    expect = func(input)
    res = compiled(input)
    assert is_similar(expect, res)


class TestFoldCat:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_foldcatcat_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs():
            cat_cat_test(self.xpu_graph, pattern_func, dynamic)
            assert "Pattern.FoldCatCat changed graph" in caplog.text

    @pytest.mark.parametrize(
        "pattern_func",
        [fn1, fn2, fn2_xfail, fn2_xfail2],
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_cat_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs():
            cat_test(self.xpu_graph, pattern_func, dynamic)
            if "xfail" in pattern_func.__name__:
                assert "Pattern.FoldCat changed graph" not in caplog.text
            else:
                assert "Pattern.FoldCat changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    cat_cat_test(xpu_graph, fn0)
    cat_test(xpu_graph, fn1)
