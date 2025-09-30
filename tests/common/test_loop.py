import torch

import xpu_graph
from xpu_graph import XpuGraph, XpuGraphConfig


def loop_fn(input):
    result = 0
    for _ in range(5):
        input = input.squeeze().reshape(-1, 1)
        result = result + input
    return result


def test_cse():
    compiler = XpuGraph(XpuGraphConfig(is_training=False))
    for _p in compiler._pass_manager._passes:
        if isinstance(_p, xpu_graph.passes.cse.Cse):
            compiler._pass_manager._passes.remove(_p)
            break

    compiled_func = torch.compile(loop_fn, backend=compiler, dynamic=False)
    compiled_func(torch.randn(100, 100))


if __name__ == "__main__":
    test_cse()
