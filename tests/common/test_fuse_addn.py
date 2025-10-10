import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache


def fusedadd_4_operands(x1, x2, x3, x4):
    """Test case with 4 operands - should be fused"""
    a = x1 + x2
    b = a + x3
    c = b + x4
    return c


def fusedadd_4_operands_xfail(x1, x2, x3, x4):
    """Test case with 4 operands - should not be fused"""
    a = x1 + x2
    a = a + 0.1
    b = a + x3
    c = b + x4
    return c


def fusedadd_4_operands_mix_xfail(x1, x2, x3, x4):
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


def fusedadd_5_operands_mix(x1, x2, x3, x4, x5):
    """Test case with 5 operands - should be fused"""
    a = x1 + x2
    b = a + x3
    c = b + x4
    d = c + x5
    return d


def fusedadd_5_operands_rev(x1, x2, x3, x4, x5):
    """Test case with 5 operands - should be fused"""
    e = x1 + 0.1
    a = e + x2
    b = a + x3
    c = b + x4
    d = c + x5
    return d


def fusedadd_8_operands_twice(x1, x2, x3, x4, x5, x6, x7, x8):
    """Test case with 8 operands - should be fused"""
    a = x1.mean(dim=-1, keepdim=True) + x2.mean(dim=-1, keepdim=True)
    b = a + x3.mean(dim=-1, keepdim=True)
    c = b + x4.mean(dim=-1, keepdim=True)
    d = c + x5
    e = d + x6
    f = e + x7
    g = f + x8
    return g


def can_fuse_test(xpu_graph, func, args):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=None)
    for _ in range(3):
        args = [arg.repeat(2, 1) for arg in args]
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
    x6 = torch.rand(10, 20)
    x7 = torch.rand(10, 20)
    x8 = torch.rand(10, 20)

    @pytest.mark.parametrize(
        "func, args",
        [
            (fusedadd_4_operands, (x1, x2, x3, x4)),
            (fusedadd_5_operands, (x1, x2, x3, x4, x5)),
            (fusedadd_4_operands_mix_xfail, (x1, x2, x3, x4)),
            (fusedadd_5_operands_mix, (x1, x2, x3, x4, x5)),
            (fusedadd_5_operands_rev, (x1, x2, x3, x4, x5)),
            (fusedadd_8_operands_twice, (x1, x2, x3, x4, x5, x6, x7, x8)),
        ],
    )
    def test_can_fuse_case(self, caplog, func, args):
        """Test cases that should be fused"""
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            args = list(args)
            if "mix" in func.__name__:
                args[0] = args[0].to(torch.int32)
            can_fuse_test(self.xpu_graph, func, args)
        if "xfail" in func.__name__:
            assert "Pattern.FusedAddN changed graph" not in caplog.text
        else:
            assert caplog.text.count("Pattern.FusedAddN changed graph") == 2


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=False,
        opt_level=OptLevel.level2,
    )
    xpu_graph_instance = xpu_graph.compiler.XpuGraph(config)

    # Test positive cases
    x1, x2, x3, x4, x5 = [torch.rand(10, 20) for _ in range(5)]
    can_fuse_test(xpu_graph_instance, fusedadd_4_operands, [x1, x2, x3, x4])
    can_fuse_test(xpu_graph_instance, fusedadd_5_operands, [x1, x2, x3, x4, x5])
