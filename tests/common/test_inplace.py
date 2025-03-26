import torch
import torch.nn as nn
import torch.nn.functional as F
from xpu_graph.config import OptLevel
import xpu_graph
from xpu_graph.test_utils import aggregate_similar
import pytest

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0(indices, values):
    max_len = torch.max(indices) + 1
    result = torch.zeros(max_len, dtype=values.dtype, device=values.device)
    result.scatter_(0, indices, values)
    return result


def fn1(indices, values):
    values2 = values.add_(1)
    return values2


def fn2(indices, values):
    values1 = values.add(1)
    values2 = values.add_(1)
    return values1, values2


def fn3(indices, values):
    values1 = values.add_(1)
    values2 = values.clone()
    values3 = values1.add_(1)
    return values1, values2, values3


def fn4(indices, values):
    values1 = values.add_(1)
    values2 = values1.add_(1)
    return values, values1, values2


def fn5(indices, values):
    values1 = values.add_(1)
    values2 = values1.add(2)
    values3 = values2.add_(values1)
    values1.copy_(values3)
    values4 = values1 + values2
    return values1, values2, values3, values4


def fn6(indices, values):
    values1 = values.add(1)
    values2 = values.clone()
    values2.copy_(values1)
    values1.add_(2)
    values3 = values2 + values1
    return values1, values2, values3


def inplace_test(xpu_graph, func):
    indices = torch.tensor([7, 6, 5, 4, 0, 1, 2, 3], dtype=torch.int64, device=device)
    values = torch.randn([8], dtype=data_type, device=device)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    values1 = values.clone()
    res1 = func(indices, values1)
    res = compiled(indices, values)

    assert aggregate_similar(res, res1)
    assert aggregate_similar(values, values1)


class TestInplace:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2
        )
        self.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3, fn4, fn5, fn6],
    )
    def test_inplace(self, pattern_func):
        inplace_test(self.infer_backend, pattern_func)


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, debug=True
    )
    infer_backend = xpu_graph.XpuGraph(infer_config)
    inplace_test(infer_backend, fn0)
    inplace_test(infer_backend, fn1)
    inplace_test(infer_backend, fn2)
    inplace_test(infer_backend, fn3)
    inplace_test(infer_backend, fn4)
    inplace_test(infer_backend, fn5)
    inplace_test(infer_backend, fn6)
