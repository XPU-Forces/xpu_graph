import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn0(input1, input2):
    return input2 @ input1.T


def fn1(input1, input2):
    return input1 + input2


def matmul_test(xpu_graph_backend, func, dynamic):
    input1 = torch.randn((4096, 768), dtype=data_type).to(device)
    input2 = torch.randn((4096, 768), dtype=data_type).to(device)
    torch._dynamo.reset()
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=dynamic)
    res1 = compiled(input1, input2)
    res = func(input1, input2)
    is_similar(res.cpu().float(), res1.cpu().float())


class TestMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            vendor_compiler_config={"mode": "default", "cpp_wrapper": True},
        )

    @pytest.mark.parametrize(
        "pattern_func,dynamic",
        [(fn0, False), (fn1, True)],
    )
    def test_matmul_patterns(self, caplog, pattern_func, dynamic):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            matmul_test(self.xpu_graph_backend, pattern_func, dynamic)
        if pattern_func in [fn0]:
            assert "Pattern.CustomDenseLayer changed graph" in caplog.text


if __name__ == "__main__":
    xpu_compiler_backend = xpu_graph.mlu_compiler(
        is_training=False,
        opt_level=OptLevel.level2,
        vendor_compiler_config={"mode": "default", "cpp_wrapper": True},
    )
    matmul_test(xpu_compiler_backend, fn0)
