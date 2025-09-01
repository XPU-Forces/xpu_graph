import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_zero_like


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
        add_tup = (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
        )
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in add_tup]

        for add in candidates:
            inp0 = add.args[0]
            inp1 = add.args[1]
            target_val = None
            is_match = False
            if is_zero_like(inp0):
                is_match = True
                target_val = inp1
            elif is_zero_like(inp1):
                is_match = True
                target_val = inp0

            if is_match:
                if any([isinstance(s, torch.SymInt) for s in add.meta["val"].shape]):
                    # FIXME: use shape env to get the real shape
                    continue
                changed = True
                with gm.graph.inserting_before(add):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_val, add.meta)
                add.replace_all_uses_with(fold_res)
                gm.graph.erase_node(add)

        return changed
