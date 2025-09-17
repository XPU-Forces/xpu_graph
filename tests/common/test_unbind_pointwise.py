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
    a_list = a.unbind()
    return [a * 0.1 for a in a_list]


def fn2(a):
    a_list = a.unbind()
    return [a * 0.1 for a in a_list[:-1]] + [a_list[-1]]


def unbind_poi_test(xpu_graph_backend, func):
    dtype = torch.half
    a = torch.rand(8, 4, dtype=dtype, device="cpu")
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = func(a)
    res = compiled(a)
    for i in range(len(res)):
        assert is_similar(res[i].cpu().float(), res1[i].cpu().float())


class TestUnbindPoi:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
        ],
    )
    def test_unbind_poi_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            unbind_poi_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.CombineUnbindPoi changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, debug=True)
    # unbind_poi_test(xpu_graph_backend, fn0)
    # unbind_poi_test(xpu_graph_backend, fn1)
    unbind_poi_test(xpu_graph_backend, fn2)
