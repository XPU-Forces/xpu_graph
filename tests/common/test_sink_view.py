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


def func(input, bias, bin_op, act):
    y = input.reshape(4, 2, 6, 4)
    if bias is not None:
        y = bin_op(y, bias)
    if act == "silu":
        y = F.silu(y)
    elif act == "relu":
        y = F.relu(y)
    elif act == "sigmoid":
        y = F.sigmoid(y)
    return y


def sinkview_test(xpu_graph_backend, input_shape, bias_shape, bin_op, act):
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
    res = func(input, bias, bin_op, act)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(input, bias, bin_op, act)
    assertTensorsEqual(res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True)


class TestSinkView:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2))

    @pytest.mark.parametrize(
        "input_shape,bias_shape,bin_op,act",
        [
            ((8, 6, 4), None, None, "silu"),
            ((8, 6, 4), None, torch.add, "relu"),
            ((8, 6, 4), None, torch.mul, "sigmoid"),
            ((8, 6, 4), "float", aten.add.Tensor, "none"),
            ((8, 6, 4), (4,), aten.sub.Tensor, "none"),
            ((8, 6, 4), (1, 4), lambda x, y: x + y, "silu"),
            ((8, 6, 4), (6, 1), lambda x, y: x * y, "relu"),
            ((8, 6, 4), (6, 4), lambda x, y: x - y, "none"),
            ((8, 6, 4), "int", aten.div.Tensor, "silu"),
            ((8, 6, 4), "int", torch.sub, "relu"),
        ],
    )
    def test_sink_view_patterns(self, caplog, input_shape, bias_shape, bin_op, act):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            sinkview_test(self.xpu_graph_backend, input_shape, bias_shape, bin_op, act)
        assert "Pattern.SinkView changed graph" in caplog.text

    @pytest.mark.parametrize(
        "input_shape,bias_shape,bin_op,act",
        [
            ((8, 6, 4), None, torch.add, "none"),
            ((8, 6, 4), (2, 6, 4), torch.mul, "none"),
            ((8, 6, 1, 4), (6, 4), torch.div, "silu"),
            ((4, 6, 8), (1, 6, 1), torch.sub, "none"),
        ],
    )
    def test_sink_view_xfail_patterns(self, caplog, input_shape, bias_shape, bin_op, act):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            sinkview_test(self.xpu_graph_backend, input_shape, bias_shape, bin_op, act)
        assert "Pattern.SinkView changed graph" not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True))
    sinkview_test(xpu_graph_backend, (8, 6, 4), None, None, "sigmoid")
    sinkview_test(xpu_graph_backend, (8, 6, 4), "float", torch.add, "none")
    sinkview_test(xpu_graph_backend, (8, 6, 4), (2, 6, 4), torch.sub, "none")
    sinkview_test(xpu_graph_backend, (8, 6, 4), (1, 4), torch.mul, "silu")
