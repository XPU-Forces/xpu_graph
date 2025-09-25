import pytest
import torch

import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def noop_slice_0(x):
    return x[:]


def noop_slice_1(x):
    return torch.ops.aten.slice.Tensor(x, 0, 0, x.shape[0])


def unnoop_slice(x):
    return torch.ops.aten.slice.Tensor(x, 0, 0, x.shape[0] // 2)


class FoldableSliceScatter(torch.nn.Module):
    def __init__(self, dim, start, end_offset=0):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end_offset = end_offset

    @torch.no_grad()
    def forward(self, base, view):
        end = view.shape[self.dim] + self.end_offset
        result = torch.slice_scatter(base, view, self.dim, self.start, end)
        return result + view


class FoldableSliceScatterFixed(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    @torch.no_grad()
    def forward(self, base, view):
        end = view.shape[self.dim]
        result = torch.slice_scatter(base, view, self.dim, 0, end)
        return result + view


class TestFoldSlice:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False, vendor_compiler_config=None)
        self.xpu_graph_backend = xpu_graph.compiler.XpuGraph(config)

    input_tensor = torch.rand(128)

    @pytest.mark.parametrize(
        "testcase",
        [
            (noop_slice_0),
            (noop_slice_1),
        ],
    )
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_noop_slice_is_folded(self, testcase, caplog, dynamic):
        expect = testcase(self.input_tensor)

        with need_xpu_graph_logs():
            compiled_mod = torch.compile(testcase, backend=self.xpu_graph_backend, dynamic=dynamic)
            result = compiled_mod(self.input_tensor)

        assert is_similar(expect, result)
        assert "Pattern.FoldSliceLike changed graph" in caplog.text

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_unnoop_slice_is_not_folded(self, caplog, dynamic):
        mod = unnoop_slice
        expect = mod(self.input_tensor)

        with need_xpu_graph_logs():
            compiled_mod = torch.compile(mod, backend=self.xpu_graph_backend, dynamic=dynamic)
            result = compiled_mod(self.input_tensor)

        assert is_similar(expect, result)
        assert "Pattern.FoldSliceLike changed graph" not in caplog.text

    def test_dynamic_slice(self, caplog):
        expect = noop_slice_1(self.input_tensor)

        with need_xpu_graph_logs():
            compiled_mod = torch.compile(noop_slice_1, backend=self.xpu_graph_backend, dynamic=True)
            result = compiled_mod(self.input_tensor)

        assert is_similar(expect, result)
        assert "Pattern.FoldSliceLike changed graph" not in caplog.text


class TestFoldSliceScatter:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_foldable_slice_scatter(self, caplog, dim, dynamic):
        if dynamic:
            mod = FoldableSliceScatterFixed(dim=dim)
        else:
            mod = FoldableSliceScatter(dim=dim, start=0)

        base = torch.randn(8, 16, 32)
        view = torch.ones(8, 16, 32)

        expect = mod(base, view)

        with need_xpu_graph_logs():
            compiled_mod = torch.compile(mod, backend=self.xpu_graph, dynamic=dynamic)
            result = compiled_mod(base, view)

        assert is_similar(result, expect)
        assert "Pattern.FoldSliceLike changed graph" in caplog.text
