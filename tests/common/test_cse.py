import torch
from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    is_similar,
)


def test_cse(caplog):
    def can_cse(x, y):
        z = x + y
        zz = x + y
        return z + zz

    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, enable_cache=False))
    with need_xpu_graph_logs():
        compiled_func = torch.compile(
            can_cse,
            backend=xpu_graph_backend,
        )
        in1, in2 = torch.randn(10), torch.randn(10)
        res = compiled_func(in1, in2)
        expect = can_cse(in1, in2)

    assert is_similar(res, expect) and "Optimizer.Cse changed graph" in caplog.text
