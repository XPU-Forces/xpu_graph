import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu
from xpu_graph.config import OptLevel
import torch_mlu_ops
import xpu_graph
from xpu_graph.test_utils import is_similar
import pytest
from xpu_graph.test_utils import assertTensorsEqual

torch._inductor.config.comprehensive_padding=False

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten

def fn0(inputs, weight, bias, scale, q_weight, q_bias):
    inputs = inputs * scale
    origin_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=False)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=origin_dtype)
    mm = torch.matmul(outputs, q_weight)
    return mm

def fn1(inputs, weight, bias, scale, q_weight, q_bias):
    inputs = inputs * scale
    origin_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=False)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=origin_dtype)
    mm = torch.matmul(outputs, q_weight) + q_bias
    return mm

def fn2(inputs, weight, bias, scale, q_weight, q_bias):
    inputs = inputs * scale
    origin_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=False)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=origin_dtype)
    mm = torch.matmul(outputs, q_weight) + q_bias
    return mm

def layernorm_test(xpu_graph, func):
    inputs = torch.randn((64, 1362), device=device, dtype=data_type)
    residual = torch.randn((64, 1362), device=device, dtype=data_type)
    weight = torch.randn((1362), device=device, dtype=data_type)
    bias = torch.randn((1362,), device=device, dtype=data_type)
    q_weight = torch.randn((1362, 1362), device=device, dtype=data_type)
    scale = 0.1
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    if func == fn0:
        q_bias = None
        norm = compiled(inputs, weight, bias, scale, q_weight, q_bias)
        norm1 = func(inputs, weight, bias, scale, q_weight, q_bias)
        assertTensorsEqual(
            norm1.cpu().float(), norm.cpu().float(), 0.005, use_MSE=True, use_RAE=True
        )

    if func == fn1:
        q_bias = torch.randn((1362), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias, scale, q_weight, q_bias)
        norm1 = func(inputs, weight, bias, scale, q_weight, q_bias)
        assertTensorsEqual(
            norm1.cpu().float(), norm.cpu().float(), 0.005, use_MSE=True, use_RAE=True
        )

    if func == fn2:
        q_bias = torch.randn((64, 1362), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias, scale, q_weight, q_bias)
        norm1 = func(inputs, weight, bias, scale, q_weight, q_bias)
        assertTensorsEqual(
            norm1.cpu().float(), norm.cpu().float(), 0.005, use_MSE=True, use_RAE=True
        )


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [],
    )
    def test_layernrom_patterns(self, pattern_func):
        layernorm_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)
    layernorm_test(xpu_graph_backend, fn0)
    layernorm_test(xpu_graph_backend, fn1)
    #layernorm_test(xpu_graph_backend, fn2)
