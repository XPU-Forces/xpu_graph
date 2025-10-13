import pytest
import torch

import xpu_graph
from tests.common.test_models import all_models, compare_training_compile
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "cpu"
data_type = torch.float32


class TestTraining:
    def setup_class(self):
        train_config = xpu_graph.XpuGraphConfig(is_training=True, opt_level=OptLevel.level1, freeze=False)
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_training(self, ReproCls):
        compare_training_compile(device, data_type, ReproCls, self.train_backend)


class TestTrainingWithInterceptor:
    def setup_class(self):
        train_config = xpu_graph.XpuGraphConfig(
            is_training=True, opt_level=OptLevel.level1, freeze=False, enable_interceptor="rtol=1e-6,atol=1e-5"
        )
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_training(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_training_compile(device, data_type, ReproCls, self.train_backend)
        assert "Monitored forward" in caplog.text
        assert "Monitored backward" in caplog.text
        assert "diverges" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=True, opt_level=OptLevel.level1, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_training_compile(device, data_type, ModCls, xpu_graph_backend)
