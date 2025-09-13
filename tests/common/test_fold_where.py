import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def can_fold_test(xpu_graph, dynamic):
    def _can_fold(x):
        cond = torch.randn_like(x) >= 0
        return torch.where(cond, x, x)

    x = torch.randn(128, 64)
    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _can_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


def can_reduce_shape_test(xpu_graph, dynamic):
    def _can_fold(x):
        cond = x <= 0
        return torch.where(cond, torch.zeros_like(x), x)

    x = torch.randn(128, 64)
    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _can_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldWhere:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize("test_func", [can_fold_test, can_reduce_shape_test])
    def test_can_fold_case(self, caplog, test_func):
        with need_xpu_graph_logs():
            test_func(self.xpu_graph, dynamic=False)
            assert "Pattern.FoldWhere changed graph" in caplog.text


class TestFoldWhereDynamic:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=True)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize("test_func", [can_fold_test, can_reduce_shape_test])
    def test_can_fold_case(self, caplog, test_func):
        with need_xpu_graph_logs():
            test_func(self.xpu_graph, dynamic=True)
            assert "Pattern.FoldWhere changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    # can_fold_test(xpu_graph, dynamic=False)
    can_reduce_shape_test(xpu_graph, dynamic=False)
