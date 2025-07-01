import pytest

# import random
import torch

import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

device = "cpu"
data_type = torch.float16


def fn0(a):
    unsqueeze = torch.unsqueeze(a, dim=0)
    output = torch.squeeze(unsqueeze, dim=[0])
    return output


def fn1(a):
    unsqueeze = torch.unsqueeze(a, dim=0)
    output = torch.squeeze(unsqueeze, dim=0)
    return output


def fn2(a):
    unsqueeze = torch.unsqueeze(a, dim=0)
    output = torch.squeeze(unsqueeze)
    return output


def fn3(a):
    unsqueeze = torch.unsqueeze(a, dim=0)
    output = torch.squeeze(unsqueeze)
    return output


def fn4(a):
    squeeze = torch.squeeze(a)
    output = torch.squeeze(squeeze, dim=0)
    return output


def fn5(a):
    squeeze0 = torch.squeeze(a, dim=0)
    squeeze1 = torch.squeeze(squeeze0, dim=0)
    squeeze2 = torch.unsqueeze(squeeze1, dim=0)
    squeeze3 = torch.squeeze(squeeze2, dim=0)
    squeeze4 = torch.squeeze(squeeze3)
    relu = torch.relu(squeeze4)
    squeeze5 = torch.squeeze(relu, dim=0)
    squeeze6 = torch.squeeze(squeeze5, dim=0)
    squeeze7 = torch.squeeze(squeeze6)
    return squeeze7


def fn6(a):
    squeeze0 = torch.squeeze(a, dim=0)
    squeeze1 = torch.squeeze(squeeze0, dim=0)
    squeeze2 = torch.squeeze(squeeze1)
    squeeze3 = torch.squeeze(squeeze2, dim=0)
    squeeze4 = torch.squeeze(squeeze3, dim=0)
    relu = torch.relu(squeeze4)
    squeeze5 = torch.squeeze(relu, dim=0)
    squeeze6 = torch.squeeze(squeeze5, dim=0)
    squeeze7 = torch.squeeze(squeeze6)
    squeeze8 = torch.squeeze(squeeze7, dim=0)
    return squeeze8


def fn7(a):
    squeeze0 = torch.squeeze(a, dim=0)
    squeeze1 = torch.squeeze(squeeze0, dim=0)
    squeeze2 = torch.squeeze(squeeze1, dim=0)
    squeeze3 = torch.squeeze(squeeze2, dim=0)
    squeeze4 = torch.squeeze(squeeze3, dim=0)
    relu = torch.relu(squeeze4)
    squeeze5 = torch.squeeze(relu, dim=0)
    squeeze6 = torch.squeeze(squeeze5, dim=0)
    squeeze7 = torch.squeeze(squeeze6)
    squeeze8 = torch.squeeze(squeeze7, dim=0)
    return squeeze8


def fn8(a):
    squeeze0 = torch.squeeze(a, dim=0)
    squeeze1 = torch.squeeze(squeeze0, dim=0)
    squeeze2 = torch.squeeze(squeeze1)
    squeeze3 = torch.squeeze(squeeze2, dim=0)
    squeeze4 = torch.squeeze(squeeze3, dim=0)
    relu = torch.relu(squeeze4)
    squeeze5 = torch.squeeze(relu, dim=0)
    squeeze6 = torch.squeeze(squeeze5, dim=0)
    squeeze7 = torch.squeeze(squeeze6)
    squeeze8 = torch.squeeze(squeeze7, dim=0)
    return squeeze8, squeeze6


def squeeze_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    a = torch.randn(128, 64)
    if func in [fn2]:
        a = torch.randn(128, 1, 64, 1)
    if func in [fn5, fn6, fn7, fn8]:
        a = torch.randn(1, 1, 1, 1, 1, 1, 1, 256, 1, 1, 512)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].cpu().float(), res1[i].cpu().float())


class TestSqueeze:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

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
        ],
    )
    def test_squeeze_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            squeeze_test(self.xpu_graph, pattern_func)
        if pattern_func in [fn4, fn5, fn6, fn7, fn8]:
            assert "Pattern.FoldSqueeze0 changed graph" in caplog.text
        elif pattern_func in [fn5]:
            assert "Pattern.FoldSqueeze1 changed graph" in caplog.text
        else:
            assert "Pattern.FoldSqueeze1 changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    squeeze_test(xpu_graph, fn0)
    squeeze_test(xpu_graph, fn1)
    squeeze_test(xpu_graph, fn2)
    squeeze_test(xpu_graph, fn3)
    squeeze_test(xpu_graph, fn4)
    squeeze_test(xpu_graph, fn5)
    squeeze_test(xpu_graph, fn6)
    squeeze_test(xpu_graph, fn7)
    squeeze_test(xpu_graph, fn8)
