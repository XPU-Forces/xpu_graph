import pytest
import torch
import torch_mlu

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
aten = torch.ops.aten


def fn0(inputs, slice_param):
    sum_list = []
    for arg in slice_param:
        s = inputs[:, arg[0] : arg[1], :]
        a = torch.sum(s, dim=[1])
        sum_list.append(a)
    output = torch.cat(sum_list, dim=1)
    return output


def fn1(inputs, slice_param, other_tensor):
    sum_list = [other_tensor]
    for arg in slice_param:
        s = inputs[:, arg[0] : arg[1], :]
        a = torch.sum(s, dim=[1])
        sum_list.append(a)
    output = torch.cat(sum_list, dim=1)
    return output


def fn2(inputs, slice_param, other_tensor):
    sum_list = [other_tensor]
    for arg in slice_param:
        s = inputs[:, arg[0] : arg[1], :]
        a = torch.sum(s, dim=[1])
        sum_list.append(a)
    sum_list.append(other_tensor)
    output = torch.cat(sum_list, dim=1)
    return output


def fn3(inputs, slice_param, other_tensor):
    sum_list = []
    for arg in slice_param:
        s = inputs[:, arg[0] : arg[1], :]
        a = torch.sum(s, dim=[1])
        sum_list.append(a)
    other_tensor = torch.sum(other_tensor, dim=[1])
    sum_list.append(other_tensor)
    output = torch.cat(sum_list, dim=1)
    return output


def fn4(inputs, slice_param, other_tensor):
    sum_list = []
    for arg in slice_param:
        s = inputs[:, arg[0] : arg[1], :]
        a = torch.sum(s, dim=[1])
        sum_list.append(a)
    other_tensor = torch.sum(other_tensor, dim=[1])
    sum_list.append(other_tensor)
    sum_list += sum_list[1:]
    output = torch.cat(sum_list, dim=1)
    return output


def sumcat_test(xpu_graph_backend, func, batch, max_col, slice_param):
    slice_723 = torch.rand(batch, max_col, 32).to("mlu:0").to(torch.float16)
    args = (slice_723, slice_param)
    if func in [fn1, fn2]:
        other_tensor = torch.rand(batch, 32).to("mlu:0").to(torch.float16)
        args += (other_tensor,)
    elif func in [fn3, fn4]:
        other_tensor = torch.rand(batch, max_col, 32).to("mlu:0").to(torch.float16)
        args += (other_tensor,)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = func(*args)
    res = compiled(*args)
    assertTensorsEqual(res1.cpu().float(), res.cpu().float(), 0.001, use_MSE=True, use_RAE=True)


class TestSliceSumCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4],
    )
    @pytest.mark.parametrize(
        "batch,max_col,slice_param",
        [
            (86, 32, [(0, 2), (0, 4), (0, 8), (0, 16)]),
            (32, 32, [(0, 2), (0, 4), (0, 8), (0, 16)]),
            (64, 32, [(0, 2), (0, 4), (0, 8), (0, 16)]),
            (128, 32, [(0, 2), (0, 4), (0, 8), (0, 16)]),
            (256, 32, [(0, 2), (0, 4), (0, 8), (0, 16)]),
            (512, 32, [(0, 2), (0, 4), (0, 8), (0, 16)]),
            (4, 4096, [(0, 2), (0, 4), (0, 16), (0, 2048)]),
        ],
    )
    def test_slice_sum_cat_patterns(self, caplog, pattern_func, batch, max_col, slice_param):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            sumcat_test(self.xpu_graph_backend, pattern_func, batch, max_col, slice_param)
        assert "Pattern.FusedSliceSumCat changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)
    sumcat_test(xpu_graph_backend, fn0, 86, 32, [(0, 2), (0, 4), (0, 8), (0, 16)])
    sumcat_test(xpu_graph_backend, fn1, 32, 32, [(0, 2), (0, 4), (0, 8), (0, 16)])
    sumcat_test(xpu_graph_backend, fn2, 64, 32, [(0, 2), (0, 4), (0, 8), (0, 16)])
    sumcat_test(xpu_graph_backend, fn3, 128, 32, [(0, 2), (0, 4), (0, 8), (0, 16)])
    sumcat_test(xpu_graph_backend, fn4, 256, 32, [(0, 2), (0, 4), (0, 8), (0, 16)])
    sumcat_test(xpu_graph_backend, fn0, 512, 32, [(0, 2), (0, 4), (0, 8), (0, 16)])
    sumcat_test(xpu_graph_backend, fn0, 4, 4096, [(0, 2), (0, 4), (0, 16), (0, 2048)])
