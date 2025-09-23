import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def mul_scalar_one(x):
    return x * 1


def mul_tensor_one(x):
    return x * torch.ones_like(x)


def mul_tensor_dyn(x):
    return 42 * torch.ones_like(x)


def mul_tensor_int(x):
    return x.int() * torch.ones_like(x, dtype=torch.int64)


def mul_tensor_shape(x):
    return x * torch.ones_like(x.unsqueeze(-1))


def mul_tensor_seq(x):
    return x * torch.ones_like(x) * 1


def can_fold_test(xpu_graph, func, x, dynamic):
    torch._dynamo.reset()
    compiled = torch.compile(func, backend=xpu_graph, dynamic=dynamic)
    expect = func(x)
    res = compiled(x)
    assert is_similar(expect, res)


class TestFoldMul:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    x = torch.rand(100)

    @pytest.mark.parametrize(
        "func, x",
        [
            (mul_scalar_one, x),
            (mul_tensor_one, x),
            (mul_tensor_dyn, x),
            (mul_tensor_int, x),
            (mul_tensor_shape, x),
            (mul_tensor_seq, x),
        ],
    )
    def test_can_fold_case(self, caplog, func, x):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, func, x, dynamic=False)
            assert "Pattern.FoldMul1 changed graph" in caplog.text


class TestFoldMulDynamic:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=True)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    x = torch.rand(100)

    @pytest.mark.parametrize(
        "func, x",
        [
            (mul_scalar_one, x),
            (mul_tensor_one, x),
            (mul_tensor_dyn, x),
            (mul_tensor_int, x),
            (mul_tensor_shape, x),
            (mul_tensor_seq, x),
        ],
    )
    def test_can_fold_case(self, caplog, func, x):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph, func, x, dynamic=True)
            if func in [mul_tensor_dyn, mul_tensor_shape]:
                assert "Pattern.FoldMul1 changed graph" not in caplog.text
            else:
                assert "Pattern.FoldMul1 changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    can_fold_test(xpu_graph)
