import torch
import torch.nn as nn
import torch.nn.functional as F
from xpu_graph.config import OptLevel
import xpu_graph
from xpu_graph.test_utils import is_similar
import pytest

device = "cpu"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, weight, bias):
    orig_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(
        inputs, dim=-1, keepdim=True, unbiased=False
    )  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=orig_dtype)
    return outputs


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024), device=device, dtype=data_type)
    bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0:
        bias = torch.randn((1024,), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)


class TestLayerNorm:
    def setup_class(self):
        config = xpu_graph.XpuGraphConfig(
            is_training=True, opt_level=OptLevel.level2, freeze=True
        )
        self.xpu_graph_backend = xpu_graph.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_layernorm_patterns(self, pattern_func):
        layernorm_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, freeze=True, debug=True
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    layernorm_test(xpu_graph_backend, fn0)
