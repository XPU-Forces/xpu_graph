import pytest
import torch
import torch.nn as nn

import xpu_graph
from tests.common.test_models import InplaceModel, all_models
from xpu_graph import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu"
data_type = torch.float32


def compare_inference(ModCls, backend, bsz=8, input_dim=16):
    torch._dynamo.reset()
    golden = ModCls(input_dim).to(device=device, dtype=data_type).eval()
    compiled = ModCls(input_dim).to(device=device, dtype=data_type).eval()
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)
    compiled.load_state_dict(golden.state_dict())
    compiled_input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    golden_input = compiled_input.clone()
    target = torch.randn((bsz, 1), device=device, dtype=data_type)

    loss_fn = nn.MSELoss()

    with torch.inference_mode():
        rng_state = torch.random.get_rng_state()
        loss_golden = loss_fn(golden(golden_input), target)

        torch.random.set_rng_state(rng_state)
        loss_compiled = loss_fn(compiled(compiled_input), target)

    assert is_similar(compiled_input, golden_input)
    assert is_similar(loss_golden, loss_compiled)


class TestInference:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, freeze=False)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_inference(self, ReproCls):
        with skip_xpu_graph_cache(self.infer_backend):
            compare_inference(ReproCls, self.infer_backend)


class TestFreezeInference:
    def setup_class(self):
        self.freeze_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, freeze=True)
        # Warning: DO NOT create both freeze and non-freeze in the same test case,

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_freeze_inference(self, ReproCls):
        with skip_xpu_graph_cache(self.freeze_backend):
            compare_inference(ReproCls, self.freeze_backend)


class TestInferenceWithInterceptor:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(
            is_training=False, opt_level=OptLevel.level2, freeze=False, enable_interceptor="rtol=1e-6,atol=1e-5"
        )

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_inference(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            compare_inference(ReproCls, self.infer_backend)
            assert "Monitored inference" in caplog.text
            assert "diverges" not in caplog.text


class TestFreezeInferenceWithInterceptor:
    def setup_class(self):
        self.freeze_backend = xpu_graph.mlu_compiler(
            is_training=False, opt_level=OptLevel.level2, freeze=True, enable_interceptor="rtol=1e-6,atol=1e-5"
        )
        # Warning: DO NOT create both freeze and non-freeze in the same test case,

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_freeze_inference(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.freeze_backend):
            compare_inference(ReproCls, self.freeze_backend)
            assert "Monitored inference" in caplog.text
            assert "diverges" not in caplog.text


class FaultyPattern(Pattern):
    def process(self, gm: torch.fx.GraphModule) -> bool:
        changed = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.sub.Tensor
                changed = True
        return changed


class TestInferenceXFail:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(
            is_training=False, opt_level=OptLevel.level2, freeze=False, enable_interceptor="rtol=1e-6,atol=1e-5"
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
                compare_inference(ReproCls, self.infer_backend)
        assert "The inference pass diverges" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level2, freeze=True, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    for ModCls in all_models:
        compare_inference(ModCls, xpu_graph_backend)

    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level2, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    for ModCls in all_models:
        compare_inference(ModCls, xpu_graph_backend)
