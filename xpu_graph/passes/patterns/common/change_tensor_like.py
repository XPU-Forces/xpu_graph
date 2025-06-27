import torch
import torch.fx as fx
from torch import SymInt

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern


class ChangeTensorLike(Pattern):
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False
        tensor_like_map = {
            torch.ops.aten.ones_like.default: torch.ops.aten.ones.default,
            torch.ops.aten.zeros_like.default: torch.ops.aten.zeros.default,
        }
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in tensor_like_map]

        for like in candidates:
            if any([isinstance(s, SymInt) for s in like.meta["val"].shape]):
                # FIXME: use shape env to get the real shape
                continue
            changed = True
            kwargs = {"dtype": like.meta["val"].dtype, "device": like.meta["val"].device}
            if "pin_memory" in like.kwargs:
                kwargs["pin_memory"] = like.kwargs["pin_memory"]
            with gm.graph.inserting_before(like):
                tensor = gm.graph.call_function(
                    tensor_like_map[like.target],
                    args=(list(like.meta["val"].shape),),
                    kwargs=kwargs,
                )
            like.replace_all_uses_with(tensor)
            gm.graph.erase_node(like)

        return changed
