import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def div_scalar_one(x):
    return x / 1


def div_tensor_one(x):
    return x / torch.ones_like(x)


def can_fold_test(xpu_graph, func, x, dynamic):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    expect = func(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldDiv:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    x = torch.rand(100)

    @pytest.mark.parametrize("func, x", [(div_scalar_one, x), (div_tensor_one, x)])
    def test_can_fold_case(self, caplog, func, x):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, func, x, dynamic=False)
            assert "Pattern.FoldDiv1 changed graph" in caplog.text


class TestFoldDivDynamic:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=True)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    x = torch.rand(100)

    @pytest.mark.parametrize("func, x", [(div_scalar_one, x), (div_tensor_one, x)])
    def test_can_fold_case(self, caplog, func, x):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, func, x, dynamic=True)
            assert "Pattern.FoldDiv1 changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
