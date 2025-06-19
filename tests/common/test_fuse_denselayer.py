import pytest
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


def fn0(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    return output + bias if bias is not None else output


def matmul_test(xpu_graph_backend, func):
    inputs = torch.randn((128, 5897), device=device, dtype=data_type)
    weight = torch.randn((5897, 540), device=device, dtype=data_type)
    bias = torch.randn((128, 540), device=device, dtype=data_type)
    res = func(inputs, weight, bias)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, weight, bias)
    assertTensorsEqual(res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True)


class TestMatMul:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_matmul_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            matmul_test(self.train_backend, pattern_func)
        assert "Pattern.FusedMatMulAdd changed graph" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    matmul_test(infer_backend, fn0)
    train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level2, debug=True)
    train_backend = xpu_graph.XpuGraph(train_config)
    matmul_test(train_backend, fn0)
