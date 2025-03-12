import torch
from xpu_graph import XpuGraph, XpuGraphConfig
import torch.fx as fx


def test_register_pattern():
    def _add(x, y):
        z = x + y
        return z

    def matcher(x: fx.node, y: fx.node):
        return torch.ops.aten.add.Tensor(x, y)

    def replacement(x: fx.node, y: fx.node):
        return torch.ops.aten.sub.Tensor(x, y)

    xpu_graph = XpuGraph(XpuGraphConfig(is_training=False))
    xpu_graph.get_pattern_manager().register_pattern(matcher, replacement)

    compiled = torch.compile(_add, backend=xpu_graph)
    a = torch.randn(10)
    b = torch.randn(10)
    res = compiled(a, b)

    from xpu_graph.test_utils import is_similar

    assert is_similar(res, a - b)


if __name__ == "__main__":
    test_register_pattern()
