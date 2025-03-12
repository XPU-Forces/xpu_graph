import torch
import math


def test_convert_gelu():
    def _gelu0(x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) * x

    def _gelu1(x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

    from xpu_graph import XpuGraph, XpuGraphConfig

    compiled_gelu0 = torch.compile(_gelu0, backend=XpuGraph(XpuGraphConfig(is_training=False)))
    compiled_gelu1 = torch.compile(_gelu1, backend=XpuGraph(XpuGraphConfig(is_training=False)))

    from xpu_graph.test_utils import is_similar

    input = torch.randn(10)
    is_similar(compiled_gelu0(input), _gelu0(input))
    is_similar(compiled_gelu1(input), _gelu1(input))
