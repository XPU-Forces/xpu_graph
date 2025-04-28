import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def can_fold_test(xpu_graph):
    def _can_fold(x):
        y = torch.ops.aten.clone.default(x)
        return x + y

    x = torch.randn(128, 64)
    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=False)
    expect = _can_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


def cannot_fold_test(xpu_graph):
    def _cannot_fold(x):
        y = torch.ops.aten.clone.default(x)
        return y

    x = torch.randn(10, 10)
    compiled = torch.compile(_cannot_fold, backend=xpu_graph, dynamic=False)
    expect = _cannot_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldToCopy:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    def test_can_fold_case(self, caplog):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph)
            assert "Pattern.FoldClone changed graph" in caplog.text

    def test_cannot_fold_case(self, caplog):
        with need_xpu_graph_logs():
            cannot_fold_test(self.xpu_graph)
            assert "Pattern.FoldClone changed graph" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
    cannot_fold_test(xpu_graph)
