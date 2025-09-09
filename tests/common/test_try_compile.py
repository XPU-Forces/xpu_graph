import pytest
import torch

from tests.common.test_models import InplaceModel
from xpu_graph import OptLevel, XpuGraph, XpuGraphConfig
from xpu_graph.fx_utils import FxStage
from xpu_graph.interceptor import (
    patching_try_compile,
    reset_intercept_ctx,
    set_try_compile_mode,
)
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs

device = "cpu"
input_dim = 16
bsz = 8
data_type = torch.float32


class FaultyPattern(Pattern):
    def process(self, gm: torch.fx.GraphModule) -> bool:
        changed = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.sub.Tensor
                changed = True
        return changed


class TestTryCompile:
    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward, None])
    def test_try_compile_fail(self, caplog, ReproCls, stage):
        torch._dynamo.reset()

        train_backend = XpuGraph(
            XpuGraphConfig(
                is_training=True,
                opt_level=OptLevel.level2,
                freeze=False,
                enable_cache=False,
                enable_interceptor=True,
            )
        )
        if stage is not None:
            faulty_pattern = FaultyPattern()

            faulty_pattern._support_stages = [stage]
            train_backend.get_pattern_manager().register_pattern(faulty_pattern)

        with need_xpu_graph_logs():
            try_steps = 5
            test_steps = 10
            set_try_compile_mode(True)
            with patching_try_compile():
                golden = ReproCls(input_dim).to(device=device, dtype=data_type).train()
                optim_golden = torch.optim.AdamW(golden.parameters())

                compiled = ReproCls(input_dim).to(device=device, dtype=data_type).train()
                compiled.forward = torch.compile(compiled.forward, backend=train_backend, dynamic=None)
                compiled.load_state_dict(golden.state_dict())
                optim_compiled = torch.optim.AdamW(compiled.parameters())
                optim_compiled.load_state_dict(optim_golden.state_dict())

                def train_fw_bw(inputs, targets, model, optim):
                    optim.zero_grad()
                    outputs = model(inputs)
                    loss = torch.nn.functional.mse_loss(outputs, targets)
                    loss.backward()
                    optim.step()
                    return loss

                mock_inputs = torch.randn(bsz, input_dim, device=device, dtype=data_type)
                mock_target = torch.zeros(bsz, 1, device=device, dtype=data_type)
                for step in range(test_steps):
                    loss_golden = train_fw_bw(mock_inputs, mock_target, golden, optim_golden)

                    try:
                        if step < try_steps:
                            mode = "fail_error"
                        else:
                            mode = "passthrough"
                        with reset_intercept_ctx(f"rtol=1e-6,atol=1e-5,use_golden=1,mode={mode}"):
                            loss_compiled = train_fw_bw(mock_inputs, mock_target, compiled, optim_compiled)
                    except:
                        set_try_compile_mode(False)
                        loss_compiled = train_fw_bw(mock_inputs, mock_target, compiled, optim_compiled)
                    assert is_similar(loss_golden, loss_compiled)

            set_try_compile_mode(True)

        if stage is not None:
            assert "diverges" in caplog.text
        else:
            assert "diverges" not in caplog.text
            assert caplog.text.count("Monitored forward") == try_steps
            assert caplog.text.count("Monitored backward") == try_steps
