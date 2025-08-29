import torch.fx as fx

from xpu_graph.constant_manager import get_constant_manager
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer


class Dce(Optimizer):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        const_manager = get_constant_manager(gm)
        return gm.graph.eliminate_dead_code() or const_manager.remove_redundant_constants()
