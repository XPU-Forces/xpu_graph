import pytest
import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "cpu"
data_type = torch.float32


def sinkview_test2(xpu_graph_backend):
    def func(input, bias):
        bs = input.size(0)
        x = input.reshape(bs, -1, 2, 3)
        x = x + bias
        return x

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)
    for bsz in [2, 4, 8]:
        input = torch.randn((bsz, 4, 6), device=device, dtype=data_type)
        bias = torch.randn((4, 2, 3), device=device, dtype=data_type)
        res = func(input, bias)
        res1 = compiled(input, bias)
        assert is_similar(res.cpu().float(), res1.cpu().float())


class TestSymShape:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2))

    @pytest.mark.parametrize("test_func", [sinkview_test2])
    def test_sym_shape(self, caplog, test_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            test_func(self.xpu_graph_backend)
        assert caplog.text.count("xpu_graph passes complete") == 2


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, debug=True))
    sinkview_test2(xpu_graph_backend)
