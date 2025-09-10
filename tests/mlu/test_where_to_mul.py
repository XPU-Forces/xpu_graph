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


def fn0(inputs, slice_, batch):
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_5)
    where_3 = torch.where(inputs, zeros, slice_6)
    output = torch.cat([slice_1, where_0, where_1, where_2, where_3], dim=-1)
    return output, slice_1


def fn1(inputs, slice_, batch):
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = torch.where(inputs, slice_2, zeros)
    where_1 = torch.where(inputs, slice_4, zeros)
    where_2 = torch.where(inputs, slice_5, zeros)
    where_3 = torch.where(inputs, slice_6, zeros)
    output = torch.cat([slice_1, where_0, where_1, where_2, where_3], dim=-1)
    return output, slice_1


def where_to_mul_test(xpu_graph_backend, func):
    batch = 512
    random_list = random.choices([0, 1], k=batch)
    inputs = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(batch, 35149, device=device, dtype=data_type)

    res = func(inputs, slice_, batch)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, slice_, batch)
    for i in range(len(res)):
        assert torch.equal(res[i].cpu().float(), res1[i].cpu().float())


class TestWhereToMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level1, is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
        ],
    )
    def test_where_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            where_to_mul_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedWhereToMul changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level1, is_training=False, debug=True)
    where_to_mul_test(xpu_graph_backend, fn0)
    where_to_mul_test(xpu_graph_backend, fn1)
