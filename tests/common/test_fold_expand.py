import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = a.expand(128, 64)
    return output


def expand_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    a = torch.randn(128, 64)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestExpand:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_expand_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            expand_test(self.xpu_graph, pattern_func)
        assert "Pattern.FoldExpand changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    expand_test(xpu_graph, fn0)
