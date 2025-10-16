import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = torch.stack([a], dim=0)
    return output


def fn0_strided(input):
    x = input.T
    x = torch.stack([x], dim=0)
    x = x.view(-1)
    return x


def fn1(a):
    output = torch.stack([a], dim=1)
    return output


def fn2(a):
    output = torch.stack([a], dim=2)
    return output


def fn3(a):
    outputs = a.unbind()
    output = torch.stack(outputs)
    return output


def fn4(a):
    outputs = a.unbind(dim=1)
    output = torch.stack(outputs, dim=1)
    return output


def fn4_xfail(a):
    outputs = a.unbind(dim=1)
    output = torch.stack(outputs[:4], dim=1)
    return output


def stack_test(xpu_graph, func, dynamic):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    a = torch.randn(128, 64)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestStack:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn0_strided,
            fn1,
            fn2,
            fn3,
            fn4,
            fn4_xfail,
        ],
    )
    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    def test_stack_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            stack_test(self.xpu_graph, pattern_func, dynamic)
        if "xfail" in pattern_func.__name__:
            assert "Pattern.FoldStack changed graph" not in caplog.text
        else:
            assert "Pattern.FoldStack changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    stack_test(xpu_graph, fn0)
    stack_test(xpu_graph, fn1)
    stack_test(xpu_graph, fn2)
    stack_test(xpu_graph, fn3)
    stack_test(xpu_graph, fn4)
