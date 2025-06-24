import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    is_similar,
    maybe_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fusedadd_4_operands(x1, x2, x3, x4):
    """Test case with 4 operands - should be fused"""
    a = x1 + x2
    b = a + x3
    c = b + x4
    return c


def fusedadd_5_operands(x1, x2, x3, x4, x5):
    """Test case with 5 operands - should be fused"""
    a = x1 + x2
    b = a + x3
    c = b + x4
    d = c + x5
    e = d + 0.1
    return e


def can_fuse_test(xpu_graph, func, *args):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    expect = func(*args)
    res = compiled(*args)
    assert is_similar(expect, res)


class TestFusedAddN:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(
            is_training=False,
            opt_level=OptLevel.level2,
        )
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    # Test data
    x1 = torch.rand(10, 20)
    x2 = torch.rand(10, 20)
    x3 = torch.rand(10, 20)
    x4 = torch.rand(10, 20)
    x5 = torch.rand(10, 20)

    @pytest.mark.parametrize(
        "func, args",
        [
            (fusedadd_4_operands, (x1, x2, x3, x4)),
            (fusedadd_5_operands, (x1, x2, x3, x4, x5)),
        ],
    )
    def test_can_fuse_case(self, caplog, func, args):
        """Test cases that should be fused"""
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            can_fuse_test(self.xpu_graph, func, *args)
        assert "Pattern.FusedAddN changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=False,
        opt_level=OptLevel.level2,
    )
    xpu_graph_instance = xpu_graph.compiler.XpuGraph(config)

    # Test positive cases
    x1, x2, x3, x4, x5 = [torch.rand(10, 20) for _ in range(5)]
    can_fuse_test(xpu_graph_instance, fusedadd_4_operands, x1, x2, x3, x4)
    can_fuse_test(xpu_graph_instance, fusedadd_5_operands, x1, x2, x3, x4, x5)
