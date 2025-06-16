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
data_type = torch.float32


def fn0(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight in weight_list:
        outputs_original.append(inputs @ weight)
    return outputs_original


def fn1(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight in weight_list:
        outputs_original.append(torch.bmm(inputs, weight))
    return outputs_original


def fn2(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(inputs @ weight + bias)
    return outputs_original


def fn3(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(inputs @ weight + bias)
    return outputs_original


def fn4(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(inputs @ weight + bias)
    return outputs_original


def fn5(inputs, weight_list, bias_list):
    inputs = inputs.squeeze(0)
    outputs_original = []
    for weight, bias in zip(weight_list, bias_list):
        outputs_original.append(torch.relu(inputs @ weight + bias))
    return outputs_original


def fn6(inputs, weight_list, bias_list):
    input_list = inputs.split(inputs.shape[0] // len(weight_list), dim=0)
    input_list = [i.squeeze(0) for i in input_list]
    outputs_original = []
    for input, weight in zip(input_list, weight_list):
        outputs_original.append(torch.relu(input @ weight))
    return outputs_original


def combine_matmul_test(xpu_graph_backend, func):
    T = 8
    inputs = None
    weight_list = None
    bias_list = None
    if func in [fn0, fn2, fn4, fn5]:
        M, N, K = 5, 8, 7
        inputs = torch.randn((1, M, N), device=device, dtype=data_type)
        weight_list = [torch.randn((N, K), device=device, dtype=data_type) for _ in range(T)]
        if func == fn4:
            bias_list = [torch.randn((K), device=device, dtype=data_type) for _ in range(T)]
        else:
            bias_list = [torch.randn((M, K), device=device, dtype=data_type) for _ in range(T)]
    if func in [fn1, fn3]:
        S = 3
        M, N, K = 5, 6, 7
        inputs = torch.randn((S, M, N), device=device, dtype=data_type)
        weight_list = [torch.randn((S, N, K), device=device, dtype=data_type) for _ in range(T)]
        bias_list = [torch.randn((S, M, K), device=device, dtype=data_type) for _ in range(T)]
    if func == fn6:
        M, N, K = 5, 8, 7
        inputs = torch.randn((T, M, N), device=device, dtype=data_type)
        weight_list = [torch.randn((N, K), device=device, dtype=data_type) for _ in range(T)]
        bias_list = None
    res = func(inputs, weight_list, bias_list)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, weight_list, bias_list)
    for i in range(T):
        assertTensorsEqual(
            res[i].cpu().float(),
            res1[i].cpu().float(),
            0.005,
            use_MSE=True,
            use_RAE=True,
        )


class TestCombineMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)

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
        ],
    )
    def test_matmul_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_matmul_test(self.xpu_graph_backend, pattern_func)
            assert "Pattern.FusedCombineMatMul changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, debug=False)
    combine_matmul_test(xpu_graph_backend, fn0)
    combine_matmul_test(xpu_graph_backend, fn1)
    combine_matmul_test(xpu_graph_backend, fn2)
    combine_matmul_test(xpu_graph_backend, fn3)
    combine_matmul_test(xpu_graph_backend, fn4)
    combine_matmul_test(xpu_graph_backend, fn5)
    combine_matmul_test(xpu_graph_backend, fn6)
