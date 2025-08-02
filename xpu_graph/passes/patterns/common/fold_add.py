import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.pattern_matcher import (
    FxCapture,
    add_like,
    zero_like,
)


class FoldAdd0(Pattern):
    """
    Fold aten.add(x, zero_like) -> x
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False

        src_node = FxCapture.symbol("src")

        add0_pat = add_like(src_node, zero_like()) | add_like(src_node, 0) | add_like(src_node, 0.0)

        changed = False
        for node in reversed(gm.graph.nodes):
            ctx = add0_pat.try_capture(node)
            if ctx.status:
                changed = True
                with gm.graph.inserting_before(node):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, ctx["src"], node.meta)
                node.replace_all_uses_with(fold_res)
                gm.graph.erase_node(node)

        return changed
