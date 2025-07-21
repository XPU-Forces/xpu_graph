import pytest
import torch
import torch.nn.functional as F
import torch_mlu

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn0(input1, input2):
    return input1 + input2


def matmul_test(xpu_graph_backend, func):
    input1 = torch.randn((4096, 768), dtype=data_type).to(device)
    input2 = torch.randn((4096, 768), dtype=data_type).to(device)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(input1, input2)
    res = func(input1, input2)
    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


class TestMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            vendor_compiler_config={"mode": "default", "cpp_wrapper": True},
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_matmul_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            matmul_test(self.xpu_graph_backend, pattern_func)
        if pattern_func in [fn0]:
            assert "Pattern.FusedMatMul changed graph" in caplog.text


if __name__ == "__main__":
    xpu_compiler_backend = xpu_graph.mlu_compiler(
        is_training=False,
        opt_level=OptLevel.level2,
        vendor_compiler_config={"mode": "default", "cpp_wrapper": True},
    )
    matmul_test(xpu_compiler_backend, fn0)
