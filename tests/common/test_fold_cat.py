import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import is_similar


def fn0(inputs):
    a, b, c = inputs
    a = torch.cat([a, b], dim=1)
    output = torch.cat([a, c], dim=1)
    return output


def cat_test(xpu_graph, func):
    a = torch.randn(128, 64)
    b = torch.randn(128, 32)
    c = torch.randn(128, 300)
    args = [a, b, c]
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res1 = func(args)
    res = compiled(args)
    assert is_similar(res1.float(), res.float())


class TestCat:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_sumcat_patterns(self, pattern_func):
        cat_test(self.xpu_graph, pattern_func)


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    cat_test(xpu_graph, fn0)
