import pytest
import torch
import torch_mlu
import xpu_graph

import torch.nn.functional as F
from xpu_graph.test_utils import is_similar

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn0(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    return output + bias if bias is not None else output


def fn1(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight).reshape(128, 32, 16)
    return output + bias if bias is not None else output


def fn2(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight.transpose(0, 1))
    return output + bias if bias is not None else output


def fn3(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight.transpose(0, 1)).reshape(128, 32, 16)
    return output + bias if bias is not None else output


def fn4(inputs, weight, bias):
    return fn0(inputs, weight, bias)


def fn5(inputs, weight, bias):
    return fn0(inputs, weight, bias)


def fn6(inputs, weight, bias):
    return fn0(inputs, weight, bias)


def fn7(inputs, weight, bias):
    return fn0(inputs, weight, bias)


def fn8(inputs, weight, bias):
    return fn0(inputs, weight, bias)


def fn9(inputs, weight, bias):
    return fn1(inputs, weight, bias)


def fn10(inputs, weight, bias):
    return fn2(inputs, weight, bias)


def fn11(inputs, weight, bias):
    return fn3(inputs, weight, bias)


def fn12(inputs, weight, bias):
    return torch.relu(fn3(inputs, weight, bias))


def fn13(inputs, weight, bias):
    return F.silu(fn3(inputs, weight, bias))


def fn14(inputs, weight, bias):
    return F.silu(fn3(inputs, weight, bias)).view(-1)


def fn15(inputs, weight, bias):
    return F.silu(fn3(inputs, weight, bias).reshape(128, 32, 16)).view(-1)


def fn16(inputs, weight, bias):
    return (
        F.silu(fn3(inputs, weight, bias).reshape(128, 32, 16))
        .view(-1)
        .reshape(128, 32, 16)
    )


def fn17(inputs, weight, bias):
    output = fn3(inputs, weight, bias).reshape(128, 32, 16)
    output = F.silu(output)
    return output.view(-1).reshape(128, 32, 16)


def matmul_test(xpu_graph_backend, func):
    if func in [fn0, fn1, fn9]:
        inputs = torch.randn((4096, 768), device=device, dtype=data_type)
        weight = torch.randn((768, 16), device=device, dtype=data_type)
        bias = torch.randn((16), device=device, dtype=data_type)
    elif func in [fn2, fn3, fn10, fn11, fn12, fn13, fn14, fn15, fn16]:
        inputs = torch.randn((4096, 768), device=device, dtype=data_type)
        weight = torch.randn((16, 768), device=device, dtype=data_type)
        bias = torch.randn((16), device=device, dtype=data_type)
    elif func == fn4:
        inputs = torch.randn((128, 5897), device=device, dtype=data_type)
        weight = torch.randn((5897, 540), device=device, dtype=data_type)
        bias = torch.randn((128, 540), device=device, dtype=data_type)
    elif func in [fn5, fn6, fn7, fn8]:
        inputs = torch.randn((128, 444), device=device, dtype=data_type)
        weight = torch.randn((444, 444), device=device, dtype=data_type)
        if func == fn5:
            bias = torch.randn((1, 444), device=device, dtype=data_type)
        elif func == fn6:
            bias = 1
        elif func == fn7:
            bias = torch.randn((1), device=device, dtype=data_type)
        elif func == fn8:
            bias = torch.randn((444), device=device, dtype=data_type)
    if func in [fn0, fn1, fn2, fn3, fn12]:
        bias = None
    res = func(inputs, weight, bias)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, weight, bias)
    assert is_similar(res1.float(), res.float())


class TestMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler()

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
            fn5,
            fn6,
            fn7,
            fn8,
            fn9,
            fn10,
            fn11,
            fn12,
            fn13,
            fn14,
            fn15,
            fn16,
        ],
    )
    def test_matmul_patterns(self, pattern_func):
        matmul_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    pytest.main()
