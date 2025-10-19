import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def can_fold_test(xpu_graph):
    def _can_fold(x):
        y = torch.ops.aten.clone.default(x)
        return x + y

    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=None)

    for bs in [8, 20, 80]:
        x = torch.randn(bs, 64)
        expect = _can_fold(x)
        res = compiled(x)
        assert is_similar(expect, res)


def cannot_fold_test(xpu_graph):
    def _cannot_fold(x):
        y = torch.ops.aten.clone.default(x)
        return y

    compiled = torch.compile(_cannot_fold, backend=xpu_graph, dynamic=None)
    for bs in [8, 20, 80]:
        x = torch.randn(bs, 64)
        expect = _cannot_fold(x)
        res = compiled(x)
        assert is_similar(expect, res)


def cannot_fold_test1(xpu_graph):
    def _can_fold(x):
        y = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
        return x + y

    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=None)

    for bs in [8, 20, 80]:
        x = torch.randn(bs, 4, 4, 16).contiguous(memory_format=torch.channels_last)
        expect = _can_fold(x)
        res = compiled(x)
        assert is_similar(expect, res)


def cannot_fold_test2(xpu_graph):
    def _can_fold(x):
        y = x.transpose(-2, -1)
        y = torch.ops.aten.clone.default(y, memory_format=torch.contiguous_format)
        y = y.view(-1, 16, 64)
        return x + y

    compiled = torch.compile(_can_fold, backend=xpu_graph, dynamic=None)

    for bs in [8, 20, 100]:
        x = torch.randn(16, 64, bs).permute(2, 0, 1)
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
            cannot_fold_test1,
            cannot_fold_test2,
        ],
    )
    def test_can_fold_case(self, caplog, test_func):
        with need_xpu_graph_logs():
            test_func(self.xpu_graph)
            if test_func.__name__.startswith("can_fold"):
                assert caplog.text.count("Pattern.FoldClone changed graph") == 2
            else:
                assert "Pattern.FoldClone changed graph" not in caplog.text
            assert caplog.text.count("xpu_graph passes start FxStage.inference") == 2


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
    cannot_fold_test(xpu_graph)
    cannot_fold_test2(xpu_graph)
