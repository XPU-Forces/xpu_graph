import pytest
import torch
import torch.fx as fx

from xpu_graph import OptLevel, XpuGraph, XpuGraphConfig


def register_pattern_test(backend_level, pattern_level):
    def _add(x, y):
        z = x + y
        return z

    def matcher(x: fx.node, y: fx.node):
        return torch.ops.aten.add.Tensor(x, y)

    def replacement(x: fx.node, y: fx.node):
        return torch.ops.aten.sub.Tensor(x, y)

    if backend_level is None:
        xpu_graph = XpuGraph(XpuGraphConfig(is_training=False))
    else:
        xpu_graph = XpuGraph(XpuGraphConfig(is_training=False, opt_level=backend_level))
    if pattern_level is None:
        xpu_graph.get_pattern_manager().register_pattern(matcher, replacement)
    else:
        xpu_graph.get_pattern_manager().register_pattern(matcher, replacement, opt_level=pattern_level)

    compiled = torch.compile(_add, backend=xpu_graph)
    a = torch.randn(10)
    b = torch.ones_like(a)
    res = compiled(a, b)

    from xpu_graph.test_utils import is_similar

    assert is_similar(res, a - b)

    for pass_ in xpu_graph._pass_manager._passes:
        assert pass_._opt_level == xpu_graph._config.opt_level
    for _, patts_ in xpu_graph._pass_manager._pattern_manager._patterns.items():
        for patt_ in patts_:
            assert patt_._opt_level == xpu_graph._config.opt_level


def test_register_pattern_default():
    register_pattern_test(None, None)


def test_register_pattern_update():
    register_pattern_test(OptLevel.level2, OptLevel.level0)


def test_register_pattern_xfail():
    with pytest.raises(AssertionError):
        register_pattern_test(OptLevel.level0, OptLevel.level1)


if __name__ == "__main__":
    test_register_pattern_default()
    test_register_pattern_update()
