import math

import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    a_list = a.split(2)
    return [a * 0.1 for a in a_list]


def fn1(a):
    a_list = a.split(2)
    return [a * 0.1 for a in a_list] + [a + 0.1 for a in a_list]


def split_poi_test(xpu_graph_backend, func):
    dtype = torch.half
    a = torch.rand(8, 4, dtype=dtype, device="mlu")
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = func(a)
    res = compiled(a)
    for i in range(len(res)):
        assert is_similar(res[i].cpu().float(), res1[i].cpu().float())


class TestSplitPoi:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            # fn1,
        ],
    )
    def test_mul_sum_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            split_poi_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedMulSumCat changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, debug=True)
    # split_poi_test(xpu_graph_backend, fn0)
    split_poi_test(xpu_graph_backend, fn1)
