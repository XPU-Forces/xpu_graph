import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = a.view(128, 64)
    return output


def fn1(a):
    a_view = a.view(1, -1)
    output = a_view.view(-1, 64)
    return output


def fn2(a):
    a_squeeze = a.squeeze()
    output = a_squeeze.view(-1, 1)
    return output


def fn3(a):
    a_squeeze = a.squeeze()
    output = a_squeeze.view(-1, 1)
    return a_squeeze, output


def fn4(a):
    a_squeeze = a.squeeze().view(1, -1).unsqueeze(dim=0)
    output = a_squeeze.view(-1, 1)
    return output


def fn5(a):
    a_squeeze = a.squeeze(1)
    output = a_squeeze.view(-1, 1)
    return output


def fn6(a):
    a_squeeze = a.squeeze(1, 3)
    output = a_squeeze.view(-1, 1)
    return output


def fn7(a):
    output = a.view(a.size(0), a.size(1))
    return output


def view_test(xpu_graph, func, dynamic):
    torch._dynamo.reset()
    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    a = torch.randn(128, 64)
    if func in [fn2, fn5, fn6]:
        a = torch.randn(128, 1, 64, 1)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestView:
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
            fn5,
            fn6,
            fn7,
        ],
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_view_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            view_test(self.xpu_graph, pattern_func, dynamic)
        if pattern_func in [fn0, fn7]:
            assert "Pattern.FoldView0 changed graph" in caplog.text
        else:
            assert "Pattern.FoldView1 changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    view_test(xpu_graph, fn0)
    view_test(xpu_graph, fn1)
    view_test(xpu_graph, fn2)
    view_test(xpu_graph, fn3)
    view_test(xpu_graph, fn4)
    view_test(xpu_graph, fn5)
    view_test(xpu_graph, fn6)
