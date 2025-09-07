import pytest
import torch

import xpu_graph
from tests.common.test_models import all_models, compare_inference_compile
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu"
data_type = torch.float32


class TestInference:
    def setup_class(self):
        self.infer_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, freeze=False)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_inference(self, ReproCls):
        with skip_xpu_graph_cache(self.infer_backend):
            compare_inference_compile(device, data_type, ReproCls, self.infer_backend)


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
            compare_inference_compile(device, data_type, ReproCls, self.freeze_backend)


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
            compare_inference_compile(device, data_type, ReproCls, self.infer_backend)
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
            compare_inference_compile(device, data_type, ReproCls, self.freeze_backend)
            assert "Monitored inference" in caplog.text
            assert "diverges" not in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level2, freeze=True, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    for ModCls in all_models:
        compare_inference_compile(device, data_type, ModCls, xpu_graph_backend)

    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level2, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    for ModCls in all_models:
        compare_inference_compile(device, data_type, ModCls, xpu_graph_backend)
