import pytest
import torch

aten = torch.ops.aten

import xpu_graph
from tests.common.test_models import (
    InplaceModel,
    compare_inference,
    compare_training,
    compare_training_compile,
)
from xpu_graph import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.interceptor import OpInterceptor, reset_intercept_ctx
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.test_utils import need_xpu_graph_logs

device = "mlu"
data_type = torch.float32
input_dim = 16


def test_op_monitor(caplog):
    def slow_add_tensor(a: torch.Tensor, b: torch.Tensor):
        a_np = a.numpy(force=True)
        b_np = b.numpy(force=True)
        c_np = a_np + b_np
        return torch.from_numpy(c_np)

    with need_xpu_graph_logs():
        with OpInterceptor({aten.add.Tensor: slow_add_tensor}, check_device=False, check_dtype=False):
            x = torch.randn((2, 2), device=device, dtype=data_type)
            y = torch.randn((2, 2), device=device, dtype=data_type)
            c = x + y
            print(c)

    assert f"Monitored op: {aten.add.Tensor}" in caplog.text and "diverges" not in caplog.text


def test_op_monitor_fail(caplog):
    with need_xpu_graph_logs():
        with OpInterceptor({aten.add.Tensor: aten.sub.Tensor}, check_device=False, check_dtype=False):
            x = torch.randn((2, 2), device=device, dtype=data_type)
            y = torch.randn((2, 2), device=device, dtype=data_type)
            c = x + y
            print(c)

    assert f"Monitored op: {aten.add.Tensor}" in caplog.text and "diverges" in caplog.text


class FaultyPattern(Pattern):
    def process(self, gm: torch.fx.GraphModule) -> bool:
        changed = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.sub.Tensor
                changed = True
        return changed


class TestInferenceInterceptor:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor=True,
        )
        self.faulty_pattern = FaultyPattern()
        self.infer_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_use_golden(self, caplog, ReproCls):
        torch._dynamo.reset()
        mod_golden = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=self.infer_backend, dynamic=False)
        mod_compiled.load_state_dict(mod_golden.state_dict())
        with need_xpu_graph_logs():
            with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=1"):
                compare_inference(device, data_type, mod_golden, mod_compiled)
        assert "The inference pass diverges" in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_use_actual(self, caplog, ReproCls):
        torch._dynamo.reset()
        mod_golden = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=self.infer_backend, dynamic=False)
        mod_compiled.load_state_dict(mod_golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(AssertionError):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0"):
                    compare_inference(device, data_type, mod_golden, mod_compiled)
        assert "The inference pass diverges" in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_passthrough(self, caplog, ReproCls):
        torch._dynamo.reset()
        mod_golden = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=self.infer_backend, dynamic=False)
        mod_compiled.load_state_dict(mod_golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(AssertionError):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=passthrough"):
                    compare_inference(device, data_type, mod_golden, mod_compiled)
        assert "The inference pass diverges" not in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_fallback(self, caplog, ReproCls):
        torch._dynamo.reset()
        mod_golden = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=self.infer_backend, dynamic=False)
        mod_compiled.load_state_dict(mod_golden.state_dict())
        with need_xpu_graph_logs():
            with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=fallback"):
                compare_inference(device, data_type, mod_golden, mod_compiled)
        assert "The inference pass diverges" not in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_fail_error(self, caplog, ReproCls):
        torch._dynamo.reset()
        mod_golden = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=self.infer_backend, dynamic=False)
        mod_compiled.load_state_dict(mod_golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(RuntimeError, match="Inference pass diverges"):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=1,mode=fail_error"):
                    compare_inference(device, data_type, mod_golden, mod_compiled)
        assert "The inference pass diverges" in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_switch_mode(self, caplog, ReproCls):
        torch._dynamo.reset()
        mod_golden = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled = ReproCls(input_dim).to(device=device, dtype=data_type).eval()
        mod_compiled.forward = torch.compile(mod_compiled.forward, backend=self.infer_backend, dynamic=False)
        mod_compiled.load_state_dict(mod_golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(RuntimeError, match="Inference pass diverges"):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=1,mode=fail_error"):
                    compare_inference(device, data_type, mod_golden, mod_compiled)
            with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=fallback"):
                compare_inference(device, data_type, mod_golden, mod_compiled)
        assert "The inference pass diverges" in caplog.text


class TestTrainingInterceptor:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor=True,
        )
        self.faulty_pattern = FaultyPattern()
        self.train_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_use_golden(self, caplog, ReproCls, stage):
        self.faulty_pattern._support_stages = [stage]
        torch._dynamo.reset()
        golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled.forward = torch.compile(compiled.forward, backend=self.train_backend, dynamic=None)
        compiled.load_state_dict(golden.state_dict())
        with need_xpu_graph_logs():
            with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=1"):
                compare_training(device, data_type, golden, compiled)
        assert (
            "The forward pass diverges" if stage == FxStage.forward else "The backward pass diverges"
        ) in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_use_actual(self, caplog, ReproCls, stage):
        self.faulty_pattern._support_stages = [stage]
        torch._dynamo.reset()
        golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled.forward = torch.compile(compiled.forward, backend=self.train_backend, dynamic=None)
        compiled.load_state_dict(golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(AssertionError):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0"):
                    compare_training(device, data_type, golden, compiled)
        assert (
            "The forward pass diverges" if stage == FxStage.forward else "The backward pass diverges"
        ) in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_passthough(self, caplog, ReproCls, stage):
        self.faulty_pattern._support_stages = [stage]
        torch._dynamo.reset()
        golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled.forward = torch.compile(compiled.forward, backend=self.train_backend, dynamic=None)
        compiled.load_state_dict(golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(AssertionError):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=passthrough"):
                    compare_training(device, data_type, golden, compiled)
        assert (
            "The forward pass diverges" if stage == FxStage.forward else "The backward pass diverges"
        ) not in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_fallback(self, caplog, ReproCls, stage):
        self.faulty_pattern._support_stages = [stage]
        torch._dynamo.reset()
        golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled.forward = torch.compile(compiled.forward, backend=self.train_backend, dynamic=None)
        compiled.load_state_dict(golden.state_dict())
        with need_xpu_graph_logs():
            with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=fallback"):
                compare_training(device, data_type, golden, compiled)
        assert (
            "The forward pass diverges" if stage == FxStage.forward else "The backward pass diverges"
        ) not in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_fail_error(self, caplog, ReproCls, stage):
        self.faulty_pattern._support_stages = [stage]
        torch._dynamo.reset()
        golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled.forward = torch.compile(compiled.forward, backend=self.train_backend, dynamic=None)
        compiled.load_state_dict(golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(
                RuntimeError, match="Forward pass diverges" if stage == FxStage.forward else "Backward pass diverges"
            ):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=fail_error"):
                    compare_training(device, data_type, golden, compiled)
        assert (
            "The forward pass diverges" if stage == FxStage.forward else "The backward pass diverges"
        ) in caplog.text

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_switch_mode(self, caplog, ReproCls, stage):
        self.faulty_pattern._support_stages = [stage]
        torch._dynamo.reset()
        golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
        compiled.forward = torch.compile(compiled.forward, backend=self.train_backend, dynamic=None)
        compiled.load_state_dict(golden.state_dict())
        with need_xpu_graph_logs():
            with pytest.raises(
                RuntimeError, match="Forward pass diverges" if stage == FxStage.forward else "Backward pass diverges"
            ):
                with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=fail_error"):
                    compare_training(device, data_type, golden, compiled)
            with reset_intercept_ctx("rtol=1e-6,atol=1e-5,use_golden=0,mode=fallback"):
                compare_training(device, data_type, golden, compiled)
        assert (
            "The forward pass diverges" if stage == FxStage.forward else "The backward pass diverges"
        ) in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, opt_level=OptLevel.level1, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )

    faulty_pattern = FaultyPattern()
    xpu_graph_backend.get_pattern_manager().register_pattern(faulty_pattern)

    faulty_pattern._support_stages = [FxStage.forward]
    compare_training_compile(device, data_type, InplaceModel, xpu_graph_backend, nsteps=2)

    faulty_pattern._support_stages = [FxStage.backward]
    compare_training_compile(device, data_type, InplaceModel, xpu_graph_backend, nsteps=2)
