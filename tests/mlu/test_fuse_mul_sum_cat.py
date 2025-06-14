import math

import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fn0(x1, x2, x3, x4):
    a = x1 * x2
    sum_a = a.sum(dim=1)
    b = x3 * x4
    sum_b = b.sum(dim=1)
    out = torch.cat([sum_a, sum_b], dim=1)
    return out


def fn1(x1, x2, x3, x4, x5, x6, x7, x8):
    a = x1 * x2
    sum_a = a.sum(dim=1)
    b = x3 * x4
    sum_b = b.sum(dim=1)
    c = x5 * x6
    sum_c = c.sum(dim=1)
    d = x7 * x8
    sum_d = d.sum(dim=1)
    out = torch.cat([sum_a, sum_b, sum_c, sum_d], dim=1)
    return out


def mul_sum_cat_test(xpu_graph_backend, func):
    batch = 1024
    dtype = torch.half
    a = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")
    b = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")
    c = torch.rand(1, 80, 32, dtype=dtype, device="mlu")
    d = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")
    e = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")
    f = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")
    g = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")
    h = torch.rand(batch, 80, 32, dtype=dtype, device="mlu")

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    if func == fn0:
        res1 = func(a, b, c, d)
        res = compiled(a, b, c, d)
    else:
        res1 = func(a, b, c, d, e, f, g, h)
        res = compiled(a, b, c, d, e, f, g, h)
    assertTensorsEqual(res.cpu().float(), res1.cpu().float(), 0.001, use_MSE=True, use_RAE=True)


class TestMulSumCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
        ],
    )
    def test_mul_sum_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            mul_sum_cat_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedMulSumCat changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
    )
    mul_sum_cat_test(xpu_graph_backend, fn0)
    mul_sum_cat_test(xpu_graph_backend, fn1)
