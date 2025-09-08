import random

import pytest
import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

device = "cpu"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, slice_, batch):
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = inputs * slice_2
    where_1 = inputs * slice_4
    where_2 = inputs * slice_5
    where_3 = inputs * slice_6
    output = torch.cat([slice_1, where_0, where_1, where_2, where_3], dim=-1)
    return output, slice_1


def fn1(inputs, slice_, batch):
    slice_1 = slice_[:, 0:4]
    slice_2 = slice_[:, 11984:11988]
    slice_3 = slice_[:, 11562:11566]
    slice_4 = slice_[:, 9782:9786]
    slice_5 = slice_[:, 9727:9731]
    slice_6 = slice_[:, 9745:9749]
    slice_7 = slice_[:, 9763:9767]
    slice_8 = slice_[:, 9773:9777]
    slice_9 = slice_[:, 9805:9809]
    slice_10 = slice_[:, 9822:9826]
    slice_11 = slice_[:, 10283:10287]
    slice_12 = slice_[:, 10736:10740]
    where_0 = slice_2 + inputs
    where_1 = slice_3 + inputs
    where_2 = slice_4 + inputs
    where_3 = slice_5 + inputs
    where_4 = slice_6 + inputs
    where_5 = slice_7 + inputs
    where_6 = slice_8 + inputs
    where_7 = slice_9 + inputs
    where_8 = slice_10 + inputs
    where_9 = slice_11 + inputs
    where_10 = slice_12 + inputs
    stack = torch.stack(
        [
            slice_1,
            where_0,
            where_1,
            where_2,
            where_3,
            where_4,
            where_5,
            where_6,
            where_7,
            where_8,
            where_9,
            where_10,
        ],
        dim=0,
    )
    return stack, slice_1


def combine_where_cat_test(xpu_graph_backend, func):
    batch = 512
    inputs = torch.randn(batch, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(batch, 35149, device=device, dtype=data_type)

    res = func(inputs, slice_, batch)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, slice_, batch)
    for i in range(len(res)):
        assert torch.equal(res[i].cpu().float(), res1[i].cpu().float())


class TestCombineWhereCat:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                opt_level=OptLevel.level1,
            )
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
        ],
    )
    def test_where_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_where_cat_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.CombinePointwiseCat changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level1, debug=True))
    combine_where_cat_test(xpu_graph_backend, fn0)
    combine_where_cat_test(xpu_graph_backend, fn1)
