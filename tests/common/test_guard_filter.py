import pytest
import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs

device = "cpu"


def foo(x):
    return x + 1


def compare_func(func, xpu_graph_backend, guard_filter_fn=None):
    guard_filter_fn = guard_filter_fn or (lambda guards: [False for _ in guards])
    compiled = torch.compile(
        func, backend=xpu_graph_backend, dynamic=False, options={"guard_filter_fn": guard_filter_fn}
    )
    for bsz in [8, 10, 12]:
        x = torch.randn(bsz, 32).to(device)
        ref = foo(x)
        y = compiled(x)
        assert is_similar(y, ref)


class TestGuardFilter:
    def setup_class(self):
        config = XpuGraphConfig(is_training=False, enable_cache=False)
        self.xpu_graph_backend = XpuGraph(config)

    @pytest.mark.parametrize(
        "func",
        [
            foo,
        ],
    )
    def test_guard_filter(self, caplog, func):
        with need_xpu_graph_logs():
            compare_func(func, self.xpu_graph_backend)

        assert caplog.text.count("xpu_graph passes start ") == 1


if __name__ == "__main__":
    config = XpuGraphConfig(is_training=False, enable_cache=False)
    xpu_graph_backend = XpuGraph(config)
    compare_func(foo, xpu_graph_backend, None)
