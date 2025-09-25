import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def where_same(x):
    cond = torch.randn_like(x) >= 0
    return torch.where(cond, x, x)


def where_one(x):
    cond = torch.randn_like(x) >= 0
    return torch.where(cond, torch.ones_like(x), torch.ones_like(x))


def where_one(x):
    cond = torch.randn_like(x) >= 0
    return torch.where(cond, torch.zeros_like(x), torch.zeros_like(x))


def where_shape(x):
    cond = torch.randn_like(x) >= 0
    return torch.where(cond, torch.zeros_like(x.unsqueeze(0)), torch.zeros_like(x.unsqueeze(1)))


def where_seq(x):
    cond = torch.randn_like(x) >= 0
    x = torch.where(cond, x, x)
    x = torch.where(cond, x, x)
    return x


def can_fold_test(xpu_graph, func, dynamic):
    x = torch.randn(128, 64)

    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    expect = func(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldWhere:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "func",
        [where_same, where_one, where_shape, where_seq],
    )
    def test_can_fold_case(self, caplog, func):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, func, dynamic=False)
            assert "Pattern.FoldWhere changed graph" in caplog.text


class TestFoldWhereDynamic:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=True)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "func",
        [where_same, where_one, where_shape, where_seq],
    )
    def test_can_fold_case(self, caplog, func):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, func, dynamic=True)
            if func in [where_shape]:
                assert "Pattern.FoldWhere changed graph" not in caplog.text
            else:
                assert "Pattern.FoldWhere changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
