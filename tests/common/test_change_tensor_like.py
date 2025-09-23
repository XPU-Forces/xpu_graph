import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = torch.ones_like(a)
    return output


def fn1(a):
    output = torch.zeros_like(a)
    return output


def fn2(a):
    output = torch.full_like(a, 42)
    return output


def fn2_dtype(a):
    output = torch.full_like(a, 42, dtype=torch.int32)
    return output


def fn2_device(a):
    output = torch.full_like(a, 42, device=torch._utils._get_available_device_type())
    return output


def inner_fn(zeros, ones):
    zeros = torch.zeros_like(zeros)
    ones = torch.ones_like(ones)
    output = torch.concat([zeros, ones], dim=0)
    return output


def fn3(a):
    b = torch.sum(a, dim=1)
    outputs = []
    for i in range(10):
        output = inner_fn(a[b >= i], a[b < i])
        outputs.append(output)
    output = torch.stack(outputs, dim=0)
    return output


def tensorlike_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=None)
    a = torch.randn(128, 64)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestTensorLike:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn2_device,
            fn2_dtype,
            fn3,
        ],
    )
    def test_tensorlike_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            tensorlike_test(self.xpu_graph, pattern_func)
        if pattern_func not in [fn3]:
            assert "Pattern.ChangeTensorLike changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    tensorlike_test(xpu_graph, fn0)
    tensorlike_test(xpu_graph, fn1)
    tensorlike_test(xpu_graph, fn2)
    tensorlike_test(xpu_graph, fn3)
