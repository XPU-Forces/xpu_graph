import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import check_op, is_one_like


class FoldDiv1(Pattern):
    """
    Fold aten.div(x, one_like) -> x
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        for div in reversed(gm.graph.nodes):
            if not check_op(div, torch.ops.aten.div.Tensor) and not check_op(div, torch.ops.aten.div.Scalar):
                continue
            inp0 = div.args[0]
            inp1 = div.args[1]
            target_val = None
            is_match = False
            if is_one_like(inp1):
                is_match = True
                target_val = inp0

            if is_match:
                with gm.graph.inserting_before(div):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_val, div.meta)

                if fold_res is not None:
                    div.replace_all_uses_with(fold_res)
                    gm.graph.erase_node(div)
                    changed = True

        return changed
