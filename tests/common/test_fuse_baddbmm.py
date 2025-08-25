import pytest
import random
import torch
import torch.nn.functional as F

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn_baddbmm(inputs, weight, bias=None):
    output = torch.bmm(inputs, weight)
    output = output + bias if bias is not None else output
    return F.gelu(output)


def bmm_test(xpu_graph_backend, func, inputs_dtype, weight_dtype, bias_dtype):
    is_training = xpu_graph_backend._config.is_training
    inputs = torch.randn((2048, 337, 16), device=device, dtype=inputs_dtype)
    weight = torch.randn((2048, 16, 16), device=device, dtype=weight_dtype, requires_grad=is_training)
    if bias_dtype in [int, float]:
        bias = bias_dtype(random.randint(1, 10))
    elif bias_dtype is None:
        bias = None
    else:
        bias = torch.randn((16), device=device, dtype=bias_dtype, requires_grad=is_training)
    res = func(inputs, weight, bias)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, weight, bias)
    assertTensorsEqual(res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True)


class TestBMM:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn_baddbmm,
        ],
    )
    @pytest.mark.parametrize(
        "is_training",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "inputs_dtype,weight_dtype,bias_dtype",
        [
            (torch.float32, torch.float32, torch.float32),
            (torch.float32, torch.float32, float),
            (torch.float32, torch.float32, None),
        ],
    )
    def test_bmm_patterns(self, caplog, pattern_func, is_training, inputs_dtype, weight_dtype, bias_dtype):
        if is_training:
            backend = self.train_backend
        else:
            backend = self.infer_backend
        with need_xpu_graph_logs(), skip_xpu_graph_cache(backend):
            bmm_test(backend, pattern_func, inputs_dtype, weight_dtype, bias_dtype)
        if bias_dtype is not None:
            assert "Pattern.FusedBAddBMM changed graph" in caplog.text
        else:
            assert "Pattern.FusedBAddBMM changed graph" not in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    bmm_test(infer_backend, fn_baddbmm, torch.float32, torch.float32, torch.float32)
    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    bmm_test(train_backend, fn_baddbmm, torch.float32, torch.float32, torch.float32)
