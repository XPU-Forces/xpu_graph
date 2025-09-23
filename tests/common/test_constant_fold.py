import pytest
import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache


def constant_folding_with_reload_test(xpu_graph_backend):
    class CanConstantFolding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(128, 128), False)

        @torch.no_grad()
        def forward(self, x):
            weight = torch.relu(torch.relu(self.weight))
            bias = torch.Tensor([1] * 128) + torch.Tensor([1])
            return torch.matmul(x + bias, weight)

    mod = CanConstantFolding()
    compiled_mod = CanConstantFolding()
    compiled_mod.load_state_dict(mod.state_dict())

    with need_xpu_graph_logs(), skip_xpu_graph_cache(xpu_graph_backend):
        compiled_mod.forward = torch.compile(compiled_mod.forward, backend=xpu_graph_backend, dynamic=False)
        res = compiled_mod(torch.ones(128, 128))
        expect = mod(torch.ones(128, 128))
        assert is_similar(res, expect)

        state_dict = {"weight": mod.weight + 1}
        mod.load_state_dict(state_dict)
        expect = mod(torch.ones(128, 128))
        compiled_mod.load_state_dict(state_dict)
        res = compiled_mod(torch.ones(128, 128))
        assert is_similar(res, expect)

        # Mock custom param reload
        state_dict = {"weight": mod.weight + 1}
        with torch.no_grad():
            for name, param in mod.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
        expect = mod(torch.ones((128, 128)))
        with torch.no_grad():
            for name, param in compiled_mod.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
        res = compiled_mod(torch.ones((128, 128)))
        assert is_similar(res, expect)


class CanConstantFolding1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(128, 128), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        weight = torch.sigmoid(torch.relu(self.weight))
        return torch.matmul(x, weight)


class CanConstantFolding2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_0 = torch.nn.Parameter(torch.rand(128, 64), requires_grad=False)
        self.weight_1 = torch.nn.Parameter(torch.rand(128, 64), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        weight = torch.cat([self.weight_0, self.weight_1], dim=1)
        return torch.matmul(x, weight)


class CanConstantFolding3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_0 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)
        self.weight_1 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        weight = self.weight_0 + self.weight_1
        return torch.matmul(x, weight)


class CanConstantFolding4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_0 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)
        self.weight_1 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)
        self.weight_2 = torch.nn.Parameter(torch.ones(128, 128), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        weight_cat = torch.cat([self.weight_0, self.weight_1], dim=1)
        return torch.matmul(torch.matmul(x, weight_cat), self.weight_2)


# --- full_like -> slice ---
class FoldFullLikeSlice(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        const_tensor = torch.full_like(torch.full_like(x, 1.0), 1.0)
        sliced_const = const_tensor[:, :64]
        return x[:, :64] + sliced_const


# --- full_like -> to ---
class FoldFullLikeTo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        const_tensor = torch.full_like(x, 2.0, dtype=torch.float32)
        casted_const = const_tensor.to(torch.int32)
        return x.to(torch.int32) * 2 + casted_const


# --- where(const_true, x, y) ---
class FoldWhereConstTrue(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, y):
        return torch.where(torch.ones(128, 128).to(torch.bool), x, y)


# --- logical_op(full_like, full_like) ---
class FoldLogicalFullFull(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        const_a = torch.full_like(x, True, dtype=torch.bool)
        const_b = torch.full_like(x, False, dtype=torch.bool)
        logical_result = torch.logical_and(const_a, const_b)
        return x * logical_result.to(x.dtype)


# --- logical_op(full_like, const) ---
class FoldLogicalFullConst(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x):
        const_a = torch.full_like(x, True, dtype=torch.bool)
        const_b = torch.full_like(x, False, dtype=torch.bool)
        logical_result = torch.logical_or(const_a, const_b)
        return x * logical_result.to(x.dtype)


class TestFreezedParamFolding:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                freeze=True,
                constant_folding=True,
                folding_freezed_params=False,
            )
        )

    def test_folding_params_enabled(self, caplog):
        constant_folding_with_reload_test(self.xpu_graph_backend)
        assert "Optimizer.ConstantFolding" in caplog.text


class TestConstantFolding:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(
            XpuGraphConfig(
                is_training=False,
                freeze=True,
                constant_folding=True,
                folding_freezed_params=True,
            )
        )

    @pytest.mark.parametrize(
        "testcase_class, inputs",
        [
            (CanConstantFolding1, (torch.rand(128, 128),)),
            (CanConstantFolding2, (torch.rand(128, 128),)),
            (CanConstantFolding3, (torch.rand(128, 128),)),
            (CanConstantFolding4, (torch.rand(128, 128),)),
            (FoldFullLikeSlice, (torch.rand(128, 128),)),
            (FoldFullLikeTo, (torch.rand(128, 128),)),
            (FoldWhereConstTrue, (torch.rand(128, 128), torch.rand(128, 128))),
            (FoldLogicalFullFull, (torch.rand(128, 128),)),
            (FoldLogicalFullConst, (torch.rand(128, 128),)),
        ],
    )
    def test_constant_folding_static(self, caplog, testcase_class, inputs):
        mod = testcase_class()
        expect = mod(*inputs)
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            torch._dynamo.reset()
            compiled_mod = torch.compile(mod, backend=self.xpu_graph_backend, dynamic=False)
            result = compiled_mod(*inputs)

        assert is_similar(result, expect), f"Failed for {testcase_class.__name__}"
        assert "Optimizer.ConstantFolding" in caplog.text
        if testcase_class == CanConstantFolding1:
            assert (
                caplog.text.count("Removed unused constant") == 1
                and "Found 1 managed constants"
                in [l for l in caplog.text.splitlines() if "managed constants in the GraphModule:" in l][-1]
            )
        elif testcase_class == CanConstantFolding2:
            assert (
                caplog.text.count("Removed unused constant") == 2
                and "Found 1 managed constants"
                in [l for l in caplog.text.splitlines() if "managed constants in the GraphModule:" in l][-1]
            )
        elif testcase_class == CanConstantFolding3:
            assert (
                caplog.text.count("Removed unused constant") == 2
                and "Found 1 managed constants"
                in [l for l in caplog.text.splitlines() if "managed constants in the GraphModule:" in l][-1]
            )
        elif testcase_class == CanConstantFolding4:
            assert (
                caplog.text.count("Removed unused constant") == 2
                and "Found 1 managed constants"
                in [l for l in caplog.text.splitlines() if "managed constants in the GraphModule:" in l][-1]
            )
        elif testcase_class == FoldLogicalFullFull:
            assert (
                caplog.text.count("Removed unused constant") == 2
                and "Found 1 managed constants"
                in [l for l in caplog.text.splitlines() if "managed constants in the GraphModule:" in l][-1]
            )
        elif testcase_class == FoldLogicalFullConst:
            assert (
                caplog.text.count("Removed unused constant") == 2
                and "Found 1 managed constants"
                in [l for l in caplog.text.splitlines() if "managed constants in the GraphModule:" in l][-1]
            )

    @pytest.mark.parametrize(
        "testcase_class, inputs",
        [
            (FoldFullLikeSlice, (torch.rand(128, 128),)),
            (FoldFullLikeTo, (torch.rand(128, 128),)),
            (FoldLogicalFullFull, (torch.rand(128, 128),)),
            (FoldLogicalFullConst, (torch.rand(128, 128),)),
        ],
    )
    def test_constant_folding_dynamic(self, caplog, testcase_class, inputs):
        mod = testcase_class()
        expect = mod(*inputs)
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            torch._dynamo.reset()
            compiled_mod = torch.compile(mod, backend=self.xpu_graph_backend, dynamic=True)
            result = compiled_mod(*inputs)

        assert is_similar(result, expect), f"Failed for {testcase_class.__name__}"
        assert "Optimizer.ConstantFolding" not in caplog.text
