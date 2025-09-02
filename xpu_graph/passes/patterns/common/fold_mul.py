import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_one_like


class FoldMul1(Pattern):
    """
    Fold aten.mul(x, one_like) -> x
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        mul_tup = (
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul.Scalar,
        )
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in mul_tup]

        for mul in candidates:
            inp0 = mul.args[0]
            inp1 = mul.args[1]
            target_val = None
            is_match = False
            if is_one_like(inp0):
                is_match = True
                target_val = inp1
            elif is_one_like(inp1):
                is_match = True
                target_val = inp0

            if is_match:
                with gm.graph.inserting_before(mul):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_val, mul.meta)

                if fold_res is not None:
                    mul.replace_all_uses_with(fold_res)
                    gm.graph.erase_node(mul)
                    changed = True

        return changed
