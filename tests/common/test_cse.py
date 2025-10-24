import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def test_cse(caplog):
    def can_cse_case_1(x, y):
        z = x + y
        zz = x + y
        return z + zz

    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, enable_cache=False))
    with need_xpu_graph_logs():
        compiled_func = torch.compile(
            can_cse_case_1,
            backend=xpu_graph_backend,
        )
        in1, in2 = torch.randn(10), torch.randn(10)
        res = compiled_func(in1, in2)
        expect = can_cse_case_1(in1, in2)

    assert is_similar(res, expect) and "Optimizer.Cse changed graph" in caplog.text

    def can_cse_case_2(x):
        return x, x.reshape(-1) + x.reshape(-1)

    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, enable_cache=False))

    with need_xpu_graph_logs():
        compiled_func = torch.compile(
            can_cse_case_2,
            backend=xpu_graph_backend,
        )
        res = compiled_func(torch.randn(10))
        expect = can_cse_case_2(torch.randn(10))

    assert "Optimizer.Cse changed graph" in caplog.text

    def can_cse_dyn_case_1(x, y):
        z = x + y
        zz = x + y
        return z + zz

    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, enable_cache=False))
    with need_xpu_graph_logs():
        compiled_func = torch.compile(
            can_cse_dyn_case_1,
            backend=xpu_graph_backend,
            dynamic=True,
        )
        in1, in2 = torch.randn(10), torch.randn(10)
        res = compiled_func(in1, in2)
        expect = can_cse_dyn_case_1(in1, in2)

    assert is_similar(res, expect) and "Optimizer.Cse changed graph" in caplog.text

    # https://github.com/pytorch/pytorch/issues/88813
    def not_cse_case_1(val: torch.Tensor):
        return val * 2, val * 2

    compiler = XpuGraph(XpuGraphConfig(is_training=False))

    compiled_func = torch.compile(not_cse_case_1, backend=compiler, dynamic=False)
    res0, res1 = compiled_func(torch.randn(100))

    assert res0.untyped_storage().data_ptr() != res1.untyped_storage().data_ptr()

    # https://github.com/pytorch/pytorch/issues/114344
    def not_cse_case_2(val: torch.Tensor):
        with torch.no_grad():
            out0 = val + 1
        out1 = val + 1
        return out0, out1

    compiled_func1 = torch.compile(not_cse_case_2, backend=compiler, dynamic=False)
    res0, res1 = compiled_func1(torch.randn(100))
    assert res0.untyped_storage().data_ptr() != res1.untyped_storage().data_ptr()

    # https://github.com/pytorch/pytorch/pull/134726#discussion_r1738774371
    def not_cse_case_3(val: torch.Tensor):
        x, y = val * 2, val * 2
        return x[0], y[0]

    compiled_func2 = torch.compile(not_cse_case_3, backend=compiler, dynamic=False)
    res0, res1 = compiled_func2(torch.randn(100))
    assert res0.untyped_storage().data_ptr() != res1.untyped_storage().data_ptr()
