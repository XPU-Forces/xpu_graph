import pytest
import torch
import torch.nn as nn

import xpu_graph
from tests.common.test_models import ConstantInplaceModel, InplaceModel, all_models
from xpu_graph import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu"
data_type = torch.float32


def compare_training(ModCls, backend, nsteps=10, bsz=8, input_dim=16):
    torch._dynamo.reset()
    golden = ModCls(input_dim).to(device=device, dtype=data_type).train()
    compiled = ModCls(input_dim).to(device=device, dtype=data_type).train()
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)
    compiled.load_state_dict(golden.state_dict())
    compiled_input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    golden_input = compiled_input.clone()
    target = torch.randn((bsz, 1), device=device, dtype=data_type)
    optimizer_golden = torch.optim.AdamW(golden.parameters())
    optimizer_compiled = torch.optim.AdamW(compiled.parameters())
    optimizer_compiled.load_state_dict(optimizer_golden.state_dict())
    print(optimizer_golden.state_dict(), optimizer_compiled.state_dict())

    loss_fn = nn.MSELoss()

    with torch.device("mlu"):
        for i in range(nsteps):
            with torch.random.fork_rng(device_type="mlu"):
                optimizer_golden.zero_grad()
                loss_golden = loss_fn(golden(golden_input), target)
                loss_golden.backward()
                optimizer_golden.step()

            optimizer_compiled.zero_grad()
            loss_compiled = loss_fn(compiled(compiled_input), target)
            loss_compiled.backward()
            optimizer_compiled.step()

            assert is_similar(compiled_input, golden_input)
            assert is_similar(loss_golden, loss_compiled)

            print(f"Step: {i} golden: {loss_golden}, compiled: {loss_compiled}")


class TestTraining:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level1,
            freeze=False,
            debuggers=["autograd"],
            vendor_compiler_config=None,  # FIXME: inductor has some bug with index_put
            cache=xpu_graph.cache.no_cache(),
        )

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_layernorm_patterns_with_loss_and_grad(self, caplog, ReproCls):
        with need_xpu_graph_logs():
            compare_training(ReproCls, self.train_backend)
        assert "Monitored forward" in caplog.text
        assert "Monitored backward" in caplog.text
        assert "diverges" not in caplog.text


class FaultyPattern(Pattern):
    def process(self, gm: torch.fx.GraphModule) -> bool:
        changed = False
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.sub.Tensor
                changed = True
        return changed


class TestTrainingXFail:
    def setup_class(self):
        self.train_backend = xpu_graph.mlu_compiler(
            is_training=True,
            opt_level=OptLevel.level1,
            freeze=False,
            debuggers=["autograd"],
            cache=xpu_graph.cache.no_cache(),
        )
        self.faulty_pattern = FaultyPattern()
        self.train_backend.get_pattern_manager().register_pattern(self.faulty_pattern)

    @pytest.mark.parametrize(
        "ReproCls",
        [InplaceModel],
    )
    @pytest.mark.parametrize("stage", [FxStage.backward, FxStage.forward])
    def test_layernorm_patterns_with_loss_and_grad(self, caplog, ReproCls, stage):
        with need_xpu_graph_logs():
            self.faulty_pattern._support_stages = [stage]
            with pytest.raises(AssertionError):
                compare_training(ReproCls, self.train_backend)
        if stage == FxStage.backward:
            assert "The backward pass diverges" in caplog.text
        else:
            assert "The forward pass diverges" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True,
        opt_level=OptLevel.level1,
        freeze=False,
        debug=True,
        debuggers=["autograd"],
    )
    for ModCls in all_models:
        compare_training(ModCls, xpu_graph_backend)

    faulty_pattern = FaultyPattern()
    xpu_graph_backend.get_pattern_manager().register_pattern(faulty_pattern)

    faulty_pattern._support_stages = [FxStage.forward]
    compare_training(InplaceModel, xpu_graph_backend, nsteps=2)

    faulty_pattern._support_stages = [FxStage.backward]
    compare_training(InplaceModel, xpu_graph_backend, nsteps=2)
