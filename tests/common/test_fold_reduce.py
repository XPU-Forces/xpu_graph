import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = torch.sum(a, dim=1)
    return output + 100


def fn1(a):
    output = torch.sum(a, dim=3, keepdim=True)
    return output


def fn2(a):
    output = torch.sum(a, dim=1, keepdim=False)
    return output


def fn3(a):
    output = torch.sum(a, dim=(1, 3), keepdim=False)
    return output


def fn4(a):
    output = torch.sum(a, dim=(1, 3), keepdim=True)
    return output


def fn5(a):
    output = torch.sum(a, dim=None, keepdim=True)
    return output


def fn6(a):
    a = torch.sum(a, dim=1, keepdim=True)
    a = torch.sum(a, dim=2, keepdim=True)
    return a


def reduce_test(xpu_graph, func, dynamic):
    torch._dynamo.reset()
    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    a = torch.randn(128, 1, 64, 1)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestReduce:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
            fn6,
        ],
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_reduce_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            reduce_test(self.xpu_graph, pattern_func, dynamic)
        assert "Pattern.FoldReduce changed graph" in caplog.text

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_reduce_none_patterns(self, caplog, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            torch._dynamo.reset()
            compiled = torch.compile(fn5, backend=self.xpu_graph, dynamic=dynamic)
            a = torch.randn(1, 1, 1, 1)
            res = fn5(a)
            res1 = compiled(a)
            for i in range(len(res)):
                assert torch.equal(res[i].float(), res1[i].float())
            assert "Pattern.FoldReduce changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    reduce_test(xpu_graph, fn0)
    # reduce_test(xpu_graph, fn1)
    # reduce_test(xpu_graph, fn2)
    # reduce_test(xpu_graph, fn3)
    # reduce_test(xpu_graph, fn4)
    # reduce_test(xpu_graph, fn5)
