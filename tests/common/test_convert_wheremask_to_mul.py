import random

import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten
dim = 32
batch = 512


def fn0(mask, slice_):
    zeros = torch.zeros([slice_.shape[0], dim], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:dim]
    slice_2 = slice_[:, 118 : (118 + dim)]
    where_1 = torch.where(mask, zeros, slice_1)
    where_2 = torch.where(mask, zeros, slice_2)
    output = where_1 + where_2
    return output


def fn1(mask, slice_):
    zeros = torch.zeros([slice_.shape[0], 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:dim]
    slice_2 = slice_[:, 118 : (118 + dim)]
    where_1 = torch.where(mask, slice_1, zeros)
    where_2 = torch.where(mask, slice_2, zeros)
    output = where_1 + where_2
    return output


def fn2(mask, slice_):
    zeros = torch.zeros([slice_.shape[0], 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 118 : (118 + dim)]
    where_1 = torch.where(mask, zeros, slice_1)
    where_2 = torch.where(mask, slice_2, zeros)
    output = where_1 + where_2
    return output


def where_to_mul_test_infer(xpu_graph_backend, func):
    random_list = random.choices([0, 1], k=batch)
    mask = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(batch, 256, device=device, dtype=data_type)

    res = func(mask, slice_)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(mask, slice_)
    assert is_similar(res.cpu().float(), res1.cpu().float())


class TestWhereToMulInfer:
    def setup_class(self):
        config = xpu_graph.XpuGraphConfig(opt_level=OptLevel.level1, is_training=False)
        self.xpu_graph_backend = xpu_graph.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2],
    )
    def test_where_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            where_to_mul_test_infer(self.xpu_graph_backend, pattern_func)
        assert "Pattern.ConvertWhereMaskToMul changed graph" in caplog.text


def where_to_mul_test_train(xpu_graph_backend, func):
    random_list = random.choices([0, 1], k=batch)
    mask = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(batch, 256, device=device, dtype=data_type, requires_grad=True)

    res = func(mask, slice_)
    loss = res.pow(2).mean()
    loss.backward()
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)

    slice_1 = slice_.clone().detach().requires_grad_()
    res1 = compiled(mask, slice_1)
    loss1 = res1.pow(2).mean()
    loss1.backward()
    assert is_similar(res.cpu().float(), res1.cpu().float())
    assert is_similar(slice_.grad.cpu().float(), slice_1.grad.cpu().float())


class TestWhereToMulTrain:
    def setup_class(self):
        config = xpu_graph.XpuGraphConfig(opt_level=OptLevel.level1, is_training=True)
        self.xpu_graph_backend = xpu_graph.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2],
    )
    def test_where_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            where_to_mul_test_train(self.xpu_graph_backend, pattern_func)
        assert "Pattern.ConvertWhereMaskToMul changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.XpuGraph(
        xpu_graph.XpuGraphConfig(opt_level=OptLevel.level1, is_training=True, debug=True)
    )
    where_to_mul_test_train(xpu_graph_backend, fn0)
