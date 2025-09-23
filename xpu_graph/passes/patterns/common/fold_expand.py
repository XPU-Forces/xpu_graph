import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern


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

        def _same_shape(org_shape, target_shape) -> bool:
            if len(org_shape) != len(target_shape):
                return False
            for os, ts in zip(org_shape, target_shape):
                if isinstance(ts, fx.Node):
                    ts = ts.meta["val"]
                if os != ts and ts != -1:
                    return False
            return True

        for expand in candidates:
            inp = expand.args[0]
            target_shape = expand.args[1]
            org_shape = list(inp.meta["val"].shape)
            if _same_shape(org_shape, target_shape):
                changed = True
                expand.replace_all_uses_with(inp)
                gm.graph.erase_node(expand)

        return changed
