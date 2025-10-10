import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import check_op, is_zero_like


class FoldSub0(Pattern):
    """
    Fold aten.sub(x, zero_like) -> x
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        sub_tup = (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
        rsub_tup = (torch.ops.aten.rsub.Tensor, torch.ops.aten.rsub.Scalar)

        for sub in reversed(gm.graph.nodes):
            if sub.op != "call_function" or sub.target not in (sub_tup + rsub_tup):
                continue
            inp0 = sub.args[0]
            inp1 = sub.args[1]
            target_val = None
            is_match = False
            if sub.target in sub_tup and is_zero_like(inp1):
                is_match = True
                target_val = inp0
            elif sub.target in rsub_tup and is_zero_like(inp0):
                is_match = True
                target_val = inp1

            if is_match:
                with gm.graph.inserting_before(sub):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_val, sub.meta)

                if fold_res is not None:
                    sub.replace_all_uses_with(fold_res)
                    gm.graph.erase_node(sub)
                    changed = True

        return changed
