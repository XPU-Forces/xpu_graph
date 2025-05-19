import pytest

import torch
import torch.nn as nn
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar


from packaging import version

torch_version = version.parse(torch.__version__[:5])

device = "npu"
data_type = torch.float32


class Simple(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        y = x + self.bias.unsqueeze(1)
        return y.sum(dim=-1)


def compare_export(ModCls, backend, bsz=8, input_dim=16, path=None):
    input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    target = torch.randn((bsz, 1), device=device, dtype=data_type)

    with torch.inference_mode():
        golden = ModCls(input_dim).to(device=device, dtype=data_type)
        exported_path = backend.aot_export(golden, (input,), package_path=path)
        exported = torch._inductor.aoti_load_package(exported_path)

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        loss_golden = loss_fn(golden(input), target)
        loss_exported = loss_fn(exported(input), target)

    assert is_similar(loss_golden, loss_exported)


@pytest.mark.skipif(
    torch_version < version.parse("2.6.0"),
    reason="Export mode not supported in torch < 2.6.0",
)
class TestInference:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2, freeze=False
        )
        self.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "ReproCls",
        [Simple],
    )
    def test_inference(self, tmp_path, ReproCls):
        compare_export(ReproCls, self.infer_backend, path=tmp_path / ReproCls.__name__)


if __name__ == "__main__":

    config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, freeze=False, debug=True
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    compare_export(Simple, xpu_graph_backend, path="test_export_debug")
