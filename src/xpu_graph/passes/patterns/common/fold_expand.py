import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.shape_utils import same_shape


class FoldExpand(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.expand.default
        ]

        for expand in candidates:
            inp = expand.args[0]
            # use target node's shape is more straightforward
            if same_shape(expand.meta["val"].shape, inp.meta["val"].shape):
                changed = True
                expand.replace_all_uses_with(inp)
                gm.graph.erase_node(expand)

        return changed
