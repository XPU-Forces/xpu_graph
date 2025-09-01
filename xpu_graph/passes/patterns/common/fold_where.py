import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_one_like, is_zero_like
from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
    get_binary_fold_result,
)


class FoldWhere(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node for node in gm.graph.nodes if node.op == "call_function" and node.target == torch.ops.aten.where.self
        ]

        for where in candidates:
            inp = where.args[1]
            other = where.args[2]
            if (
                (inp == other)
                or (is_one_like(inp) and is_one_like(other))
                or (is_zero_like(inp) and is_zero_like(other))
            ):
                if any([isinstance(s, torch.SymInt) for s in where.meta["val"].shape]):
                    # FIXME: use shape env to get the real shape
                    continue
                with gm.graph.inserting_before(where):
                    res = get_binary_fold_result(gm, inp, where.meta)
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
                    continue

        return changed
