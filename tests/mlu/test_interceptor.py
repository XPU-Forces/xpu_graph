import pytest
import torch

aten = torch.ops.aten

import xpu_graph
from tests.test_models import InplaceModel, compare_inference, compare_training
from tests.utils import parametrize_class_env
from xpu_graph import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.runtime.interceptor import OpInterceptor
from xpu_graph.test_utils import need_xpu_graph_logs

device = "mlu"
data_type = torch.float32


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


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
class TestInferenceInterceptorUseGolden:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=1",
        )
        self.faulty_pattern = FaultyPattern()
        self.infer_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_xfail_patterns(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_inference(device, data_type, ReproCls, self.infer_backend)
        assert "The inference pass diverges" in caplog.text


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
class TestInferenceInterceptorUseActual:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=0",
        )
        self.faulty_pattern = FaultyPattern()
        self.infer_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    def test_xfail_patterns(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            with pytest.raises(AssertionError):
                compare_inference(device, data_type, ReproCls, self.infer_backend)
        assert "The inference pass diverges" in caplog.text


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
class TestTrainingInterceptorUseGolden:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=1",
        )
        self.faulty_pattern = FaultyPattern()
        self.train_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_xfail_patterns(self, caplog, ReproCls, stage):
        with need_xpu_graph_logs():
            self.faulty_pattern._support_stages = [stage]
            compare_training(device, data_type, ReproCls, self.train_backend)
        if stage == FxStage.backward:
            assert "The backward pass diverges" in caplog.text
        else:
            assert "The forward pass diverges" in caplog.text


@parametrize_class_env(
    [
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "1"},
        {"XPUGRAPH_FALLBACK_LEGACY_DISPATCH": "0"},
    ],
)
class TestTrainingInterceptorUseActual:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level2,
            freeze=False,
            enable_interceptor="rtol=1e-6,atol=1e-5,use_golden=0",
        )
        self.faulty_pattern = FaultyPattern()
        self.train_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_xfail_patterns(self, caplog, ReproCls, stage):
        with need_xpu_graph_logs():
            self.faulty_pattern._support_stages = [stage]
            with pytest.raises(AssertionError):
                compare_training(device, data_type, ReproCls, self.train_backend)
        if stage == FxStage.backward:
            assert "The backward pass diverges" in caplog.text
        else:
            assert "The forward pass diverges" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, opt_level=OptLevel.level1, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )

    faulty_pattern = FaultyPattern()
    xpu_graph_backend.get_pattern_manager().register_pattern(faulty_pattern)

    faulty_pattern._support_stages = [FxStage.forward]
    compare_training(InplaceModel, xpu_graph_backend, nsteps=2)

    faulty_pattern._support_stages = [FxStage.backward]
    compare_training(InplaceModel, xpu_graph_backend, nsteps=2)
