import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def can_fold_test(xpu_graph, dynamic=True):
    def _can_fold(x):
        y = torch.ops.aten.clone.default(x)
        return x + y

    x = torch.randn(128, 64)

    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _can_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


def cannot_fold_test(xpu_graph, dynamic=True):
    def _cannot_fold(x):
        y = torch.ops.aten.clone.default(x)
        return y

    x = torch.randn(10, 10)

    compiled = torch.compile(_cannot_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _cannot_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


def cannot_fold_test2(xpu_graph, dynamic=True):
    def _can_fold(x):
        y = x.T
        y = torch.ops.aten.clone.default(y, memory_format=torch.contiguous_format)
        y = y.view(128, 64)
        return x + y

    x = torch.randn(128, 64)

    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=dynamic)
    expect = _can_fold(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldToCopy:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "test_func",
        [
            can_fold_test,
            cannot_fold_test,
            cannot_fold_test2,
        ],
    )
    @pytest.mark.parametrize(
        "dynamic",
        [
            True,
            False,
        ],
    )
    def test_can_fold_case(self, caplog, test_func, dynamic):
        with need_xpu_graph_logs():
            test_func(self.xpu_graph, dynamic)
            if test_func.__name__.startswith("can_fold"):
                assert "Pattern.FoldClone changed graph" in caplog.text
            else:
                assert "Pattern.FoldClone changed graph" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
    cannot_fold_test(xpu_graph)
    cannot_fold_test2(xpu_graph)
