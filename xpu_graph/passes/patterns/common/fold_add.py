import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_zero_like
from xpu_graph.passes.patterns.utils.pattern_matcher import (
    AnyNode,
    AtenAdd,
    ATenZeroLike,
    LiteralNode,
    NodeCapture,
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

        add_node_cap = NodeCapture()
        target_node_cap = NodeCapture()

        zeros_like_pattern = ATenZeroLike(AnyNode()) | LiteralNode(0) | LiteralNode(0.0)
        pattern = AtenAdd(AnyNode(capture=target_node_cap), zeros_like_pattern, capture=add_node_cap) | AtenAdd(
            zeros_like_pattern, AnyNode(capture=target_node_cap), capture=add_node_cap
        )

        changed = False
        for node in reversed(gm.graph.nodes):
            if pattern.match(node):
                changed = True
                with gm.graph.inserting_before(add_node_cap.node):
                    from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
                        get_binary_fold_result,
                    )

                    fold_res = get_binary_fold_result(gm, target_node_cap.node, add_node_cap.node.meta)
                add_node_cap.node.replace_all_uses_with(fold_res)
                gm.graph.erase_node(add_node_cap.node)

        return changed
