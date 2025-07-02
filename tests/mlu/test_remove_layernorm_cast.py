import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    is_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, weight, bias):
    orig_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=orig_dtype)
    return outputs


def fn1(inputs, weight, bias):
    orig_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = normalized.to(dtype=orig_dtype)
    return outputs


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024), device=device, dtype=data_type)
    bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0 or func == fn1:
        bias = torch.randn((1024,), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)


class TestLayerNormCast_forward:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level2, debug=True
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1],
    )
    def test_layernorm_cast_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            layernorm_test(self.xpu_graph_backend, pattern_func)
        assert "AutoMatchPattern.RemoveLayerNormCastForward" in caplog.text


# class TestLayerNormCast_pregrad:
#    def setup_class(self):
#        self.xpu_graph_backend = xpu_graph.mlu_compiler(
#            is_training=True, freeze=True, opt_level=OptLevel.level2, debug=True
#        )
#    @pytest.mark.parametrize(
#        "pattern_func",
#        [fn0],
#    )
#    def test_layernorm_cast_patterns(self, caplog, pattern_func):
#        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
#            layernorm_test(self.xpu_graph_backend, pattern_func)
#        assert "AutoMatchPattern.RemoveLayerNormCastPregrad" in caplog.text

if __name__ == "__main__":
    # xpu_graph_backend = xpu_graph.mlu_compiler(
    #    is_training=True, freeze=True, opt_level=OptLevel.level2, debug=True
    # )
    # layernorm_test(xpu_graph_backend, fn0)
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=True, opt_level=OptLevel.level2, debug=True)
    # layernorm_test(xpu_graph_backend, fn0)
    layernorm_test(xpu_graph_backend, fn1)
