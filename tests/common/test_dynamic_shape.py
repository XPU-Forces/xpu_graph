import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0(inputs):
    return inputs + 1


def dynamic_test(xpu_graph, func):
    torch._dynamo.reset()
    compiled = torch.compile(func, backend=xpu_graph, dynamic=None)

    for bs in range(2, 10):
        inputs = torch.randn(bs, 16)
        res1 = func(inputs)
        res = compiled(inputs)
        assert is_similar(res, res1)

    for bs in range(10, 2, -1):
        inputs = torch.randn(bs, 16)
        res1 = func(inputs)
        res = compiled(inputs)
        assert is_similar(res, res1)


class TestDynamic:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2)
        self.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_dynamic(self, pattern_func):
        dynamic_test(self.infer_backend, pattern_func)


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True)
    infer_backend = xpu_graph.XpuGraph(infer_config)
    dynamic_test(infer_backend, fn0)
