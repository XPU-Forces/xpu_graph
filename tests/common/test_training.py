import pytest
import torch

import xpu_graph
from tests.common.test_models import SimpleModel, all_models, compare_training
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs

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
        compare_training(device, data_type, ReproCls, self.train_backend)


class TestTrainingWithPartiioner:
    @pytest.mark.parametrize(
        "partition_fn",
        [
            "MINCUT",
            "DEFAULT",
            "torch._functorch.partitioners.min_cut_rematerialization_partition",
            "dummy",
        ],
    )
    def test_training(self, caplog, monkeypatch, partition_fn):
        monkeypatch.setenv("XPUGRAPH_PARTITIONER", partition_fn)

        with need_xpu_graph_logs():
            train_config = xpu_graph.XpuGraphConfig(
                is_training=True,
                opt_level=OptLevel.level1,
                freeze=False,
                enable_interceptor="rtol=1e-6,atol=1e-5",
            )
            train_backend = xpu_graph.XpuGraph(train_config)
            compare_training(device, data_type, SimpleModel, train_backend)
        assert "Monitored forward" in caplog.text
        assert "Monitored backward" in caplog.text
        assert "diverges" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=True, opt_level=OptLevel.level1, freeze=False, debug=True, enable_interceptor="rtol=1e-6,atol=1e-5"
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_training(device, data_type, ModCls, xpu_graph_backend)
