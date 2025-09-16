import pytest
import torch
import torch.nn.functional as F
import torch_mlu

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0():
    input_a = torch.randn((80, 256, 1), device=device, dtype=data_type)
    input_b = torch.randn((80, 1, 256), device=device, dtype=data_type)
    output = torch.matmul(input_a, input_b)
    return output


def fn1():
    input_a = torch.randn((80, 1, 256), device=device, dtype=data_type)
    input_b = torch.randn((80, 256, 1), device=device, dtype=data_type)
    output = torch.matmul(input_a, input_b)
    return output


def fn2():
    input_a = torch.randn((1, 256), device=device, dtype=data_type)
    input_b = torch.randn((256, 1), device=device, dtype=data_type)
    output = torch.matmul(input_a, input_b)
    return output


def bmm_test(xpu_graph_backend, func):
    res = func()
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled()
    is_similar(res.cpu().float(), res1.cpu().float())


class TestBMM:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_bmm_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            bmm_test(self.xpu_graph_backend, pattern_func)
        if pattern_func in [fn1, fn2, fn4, fn5, fn8, fn9, fn11, fn12]:
            assert "Pattern.FusedBAddBMM changed graph" in caplog.text
        else:
            assert "Pattern.FusedBAddBMM changed graph" not in caplog.text
        assert "Pattern.CustomBatchDenseLayer changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=False, opt_level=OptLevel.level2, debug=True)
    # bmm_test(xpu_graph_backend, fn0)
    # bmm_test(xpu_graph_backend, fn1)
    bmm_test(xpu_graph_backend, fn2)
