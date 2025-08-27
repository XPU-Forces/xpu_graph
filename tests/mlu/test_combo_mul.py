import random

import pytest
import torch
import torch_mlu

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(cond, args1, args2, args3, args4, args5):
    cond = cond * 5
    return cond*args1, cond*args2, cond*args3, cond*args4, cond*args5

def combo_mul_test(xpu_graph_backend, casen):
    if casen == 0:
        cond = torch.randn(1, device=device, dtype=data_type)
        a1 = torch.randn(6, device=device, dtype=data_type)
        a2 = torch.randn(2, device=device, dtype=data_type)
        a3 = torch.randn(3, device=device, dtype=data_type)
        a4 = torch.randn(4, device=device, dtype=data_type)
        a5 = torch.randn(5, device=device, dtype=data_type)
    elif casen == 1:
        cond = torch.randn(3, device=device, dtype=data_type)
        a1 = torch.randn(3, device=device, dtype=data_type)
        a2 = torch.randn(3, device=device, dtype=data_type)
        a3 = torch.randn(3, device=device, dtype=data_type)
        a4 = torch.randn(3, device=device, dtype=data_type)
        a5 = torch.randn(3, device=device, dtype=data_type)
    elif casen == 2:
        cond = torch.randn(3, device=device, dtype=data_type)
        a1 = torch.randn(1, device=device, dtype=data_type)
        a2 = torch.randn(1, device=device, dtype=data_type)
        a3 = torch.randn(1, device=device, dtype=data_type)
        a4 = torch.randn(1, device=device, dtype=data_type)
        a5 = torch.randn(1, device=device, dtype=data_type)
    elif casen == 3:
        cond = torch.randn(80,3, device=device, dtype=data_type)
        a1 = torch.randn(80,3, device=device, dtype=data_type)
        a2 = torch.randn(80,3, device=device, dtype=data_type)
        a3 = torch.randn(80,3, device=device, dtype=data_type)
        a4 = torch.randn(80,3, device=device, dtype=data_type)
        a5 = torch.randn(80,3, device=device, dtype=data_type)
    elif casen == 4:
        cond = torch.randn(80,3, device=device, dtype=data_type)
        a1 = torch.randn(80,1, device=device, dtype=data_type)
        a2 = torch.randn(80,1, device=device, dtype=data_type)
        a3 = torch.randn(80,1, device=device, dtype=data_type)
        a4 = torch.randn(80,1, device=device, dtype=data_type)
        a5 = torch.randn(80,1, device=device, dtype=data_type)
    elif casen == 5:
        cond = torch.randn(80,1, device=device, dtype=data_type)
        a1 = torch.randn(80,2, device=device, dtype=data_type)
        a2 = torch.randn(80,12, device=device, dtype=data_type)
        a3 = torch.randn(80,24, device=device, dtype=data_type)
        a4 = torch.randn(80,85, device=device, dtype=data_type)
        a5 = torch.randn(80,44, device=device, dtype=data_type)
    res = fn0(cond, a1, a2, a3, a4, a5)
    compiled = torch.compile(fn0, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(cond, a1, a2, a3, a4, a5)
    for i in range(len(res)):
        assert torch.equal(res[i].cpu().float(), res1[i].cpu().float())


class TestWhereToMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2, is_training=False, vendor_compiler_config=None)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            0,
            1,
            2,
            3,
            4,
            5,
        ],
    )
    def test_where_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combo_mul_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.ComboMulP changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2, is_training=False, debug=True, vendor_compiler_config=None)
    combo_mul_test(xpu_graph_backend, 0)
    combo_mul_test(xpu_graph_backend, 1)
    combo_mul_test(xpu_graph_backend, 2)
    combo_mul_test(xpu_graph_backend, 3)
    combo_mul_test(xpu_graph_backend, 4)
    combo_mul_test(xpu_graph_backend, 5)
