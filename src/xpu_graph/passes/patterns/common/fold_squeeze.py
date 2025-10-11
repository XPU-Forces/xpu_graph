import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_squeeze_op, check_unsqueeze_op


def match(a, b):
    return a == b if isinstance(b, int) else (len(b) == 1 and a in b)


class FoldSqueeze0(Pattern):
    """
    Fold aten.squeeze(aten.squeeze)
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False

        for node in reversed(gm.graph.nodes):
            if not check_squeeze_op(node):
                continue
            squeeze = node.args[0]
            if not check_squeeze_op(squeeze) or len(squeeze.users) > 1:
                continue
            if len(node.args) == 1:
                changed = True
                node.replace_input_with(squeeze, squeeze.args[0])
            else:
                if len(squeeze.args) == 1:
                    changed = True
                    node.replace_all_uses_with(squeeze)
                    gm.graph.erase_node(node)

        return changed


class FoldSqueeze1(Pattern):
    """
    Fold aten.squeeze(aten.unsqueeze)
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        for node in reversed(gm.graph.nodes):
            if not check_squeeze_op(node):
                continue
            unsqueeze = node.args[0]
            if not check_unsqueeze_op(unsqueeze) or len(unsqueeze.users) > 1:
                continue

            if len(node.args) == 1:
                changed = True
                node.replace_input_with(unsqueeze, unsqueeze.args[0])
            elif match(unsqueeze.args[1], node.args[1]):
                changed = True
                node.replace_all_uses_with(unsqueeze.args[0])

        return changed
