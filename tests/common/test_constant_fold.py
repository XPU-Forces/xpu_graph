import torch
from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
    is_similar,
)


def test_constant_folding(caplog):
    class CanConstantFolding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(128, 128), False)

        @torch.no_grad()
        def forward(self, x):
            weight = torch.relu(torch.relu(self.weight))
            return torch.matmul(x, weight)

    mod = CanConstantFolding()
    xpu_graph_backend = XpuGraph(XpuGraphConfig(freeze=True, is_training=False))
    with need_xpu_graph_logs(), skip_xpu_graph_cache(xpu_graph_backend):
        compiled_mod = torch.compile(mod, backend=xpu_graph_backend, dynamic=False)
        res = compiled_mod(torch.ones(128, 128))
        expect = mod(torch.ones(128, 128))

    assert (
        is_similar(res, expect)
        and "Optimizer.ConstantFolding changed graph" in caplog.text
    )
