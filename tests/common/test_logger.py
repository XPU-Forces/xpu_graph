import pytest
import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache


def demo_func(x):
    return x + 0


def test_loggers(caplog):
    with need_xpu_graph_logs(set_debug=False):
        xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, debug=False))
        compiled_func = torch.compile(demo_func, backend=xpu_graph_backend)
        x = torch.randn(2, 2)
        y = compiled_func(x)
        assert is_similar(x, y)
    assert "xpu_graph passes start FxStage.inference" in caplog.text
    assert "Pattern.FoldAdd0 changed graph" not in caplog.text

    caplog.clear()
    torch._dynamo.reset()

    with need_xpu_graph_logs(set_debug=False):
        xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, debug=True))
        compiled_func = torch.compile(demo_func, backend=xpu_graph_backend)
        x = torch.randn(2, 2)
        y = compiled_func(x)
        assert is_similar(x, y)
    assert "xpu_graph passes start FxStage.inference" in caplog.text
    assert "Pattern.FoldAdd0 changed graph" in caplog.text

    caplog.clear()
    torch._dynamo.reset()

    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, debug=False))
    with need_xpu_graph_logs(set_debug=False), skip_xpu_graph_cache(xpu_graph_backend):
        compiled_func = torch.compile(demo_func, backend=xpu_graph_backend)
        x = torch.randn(2, 2)
        y = compiled_func(x)
        assert is_similar(x, y)
    assert "xpu_graph passes start FxStage.inference" in caplog.text
    assert "Pattern.FoldAdd0 changed graph" not in caplog.text

    caplog.clear()
    torch._dynamo.reset()
