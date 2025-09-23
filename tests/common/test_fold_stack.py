import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = torch.stack([a], dim=0)
    return output


def fn1(a):
    output = torch.stack([a], dim=1)
    return output


def fn2(a):
    output = torch.stack([a], dim=2)
    return output


def fn3(a):
    a = torch.stack([a])

    return torch.stack([a])


def stack_test(xpu_graph, func, dynamic):
    torch._dynamo.reset()
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
            fn1,
            fn2,
            fn3,
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
        assert "Pattern.FoldStack changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    stack_test(xpu_graph, fn0)
    stack_test(xpu_graph, fn1)
    stack_test(xpu_graph, fn2)
