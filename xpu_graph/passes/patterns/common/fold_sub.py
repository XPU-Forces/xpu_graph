import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_zero_like


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
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in sub_tup]

        def _is_zero_like(inp) -> bool:
            scalar_tup = (
                int,
                float,
            )
            if type(inp) in scalar_tup and inp == 0:
                return True
            zero_like_tup = (
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.zeros.default,
            )
            if isinstance(inp, fx.Node) and inp.op == "call_function" and inp.target in zero_like_tup:
                return True
            return False

        for sub in candidates:
            inp0 = sub.args[0]
            inp1 = sub.args[1]
            target_val = None
            is_match = False
            if is_zero_like(inp1):
                is_match = True
                target_val = inp0

            if is_match:
                if any([isinstance(s, torch.SymInt) for s in sub.meta["val"].shape]):
                    # FIXME: use shape env to get the real shape
                    continue
                changed = True
                with gm.graph.inserting_before(sub):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_val, sub.meta)
                sub.replace_all_uses_with(fold_res)
                gm.graph.erase_node(sub)

        return changed
