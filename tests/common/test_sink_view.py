import random

import pytest
import torch
import torch.nn.functional as F

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def func(input, bias, act):
    y = input.reshape(4, 2, 6, 4)
    if bias is not None:
        y = y + bias
    if act == "silu":
        y = F.silu(y)
    elif act == "relu":
        y = F.relu(y)
    elif act == "sigmoid":
        y = F.sigmoid(y)
    return y


def sinkview_test(xpu_graph_backend, input_shape, bias_shape, act):
    torch._dynamo.reset()
    input = torch.randn(input_shape, device=device, dtype=data_type)
    if bias_shape is None:
        bias = None
    elif bias_shape == "float":
        bias = random.random()
    elif bias_shape == "int":
        bias = random.randint(0, 10)
    else:
        bias = torch.randn(bias_shape, device=device, dtype=data_type)
    res = func(input, bias, act)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(input, bias, act)
    assertTensorsEqual(res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True)


class TestSinkView:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2))

    @pytest.mark.parametrize(
        "input_shape,bias_shape,act",
        [
            ((8, 6, 4), None, "silu"),
            ((8, 6, 4), None, "relu"),
            ((8, 6, 4), None, "sigmoid"),
            ((8, 6, 4), "float", "none"),
            ((8, 6, 4), (4,), "none"),
            ((8, 6, 4), (1, 4), "silu"),
            ((8, 6, 4), (6, 1), "relu"),
            ((8, 6, 4), (6, 4), "none"),
            ((8, 6, 4), "int", "silu"),
            ((8, 6, 4), "int", "relu"),
        ],
    )
    def test_sink_view_patterns(self, caplog, input_shape, bias_shape, act):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            sinkview_test(self.xpu_graph_backend, input_shape, bias_shape, act)
        assert "Pattern.SinkView changed graph" in caplog.text

    @pytest.mark.parametrize(
        "input_shape,bias_shape,act",
        [
            ((8, 6, 4), None, "none"),
            ((8, 6, 4), (2, 6, 4), "none"),
            ((8, 6, 1, 4), (6, 4), "silu"),
            ((4, 6, 8), (1, 6, 1), "none"),
        ],
    )
    def test_sink_view_xfail_patterns(self, caplog, input_shape, bias_shape, act):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            sinkview_test(self.xpu_graph_backend, input_shape, bias_shape, act)
        assert "Pattern.SinkView changed graph" not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True))
    sinkview_test(xpu_graph_backend, (8, 6, 4), None, "sigmoid")
    sinkview_test(xpu_graph_backend, (8, 6, 4), "float", "none")
    sinkview_test(xpu_graph_backend, (8, 6, 4), (2, 6, 4), "none")
    sinkview_test(xpu_graph_backend, (8, 6, 4), (1, 4), "silu")
