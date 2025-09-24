import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def can_fold_test(xpu_graph, dynamic):
    def _can_fold(x):
        y = torch.ops.aten._to_copy.default(x)
        return x + y

    x = torch.randn(128, 64)

    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _can_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


def cannot_fold_test(xpu_graph, dynamic):
    def _cannot_fold(x):
        y = torch.ops.aten._to_copy.default(x)
        return y

    x = torch.randn(10, 10)

    compiled = torch.compile(_cannot_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _cannot_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldToCopy:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_can_fold_case(self, caplog, dynamic):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, dynamic)
            assert "Pattern.FoldToCopy changed graph" in caplog.text

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_cannot_fold_case(self, caplog, dynamic):
        with need_xpu_graph_logs():
            cannot_fold_test(self.xpu_graph, dynamic)
            assert "Pattern.FoldToCopy changed graph" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
    cannot_fold_test(xpu_graph)
