import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer


def view_to_reshape(gm: fx.GraphModule):
    """Replace every view op with reshape op"""
    changed = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in [
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
        ]:
            node.target = torch.ops.aten.reshape.default
            changed = True
    return changed


class ViewToReshape(Optimizer):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        # Note(chenyifan):
        #   reshape op may dispatch to view op if input is contiguous
        #   but the input stride may change after other passes
        #   so we need to replace every view op with reshape op
        return view_to_reshape(gm)
