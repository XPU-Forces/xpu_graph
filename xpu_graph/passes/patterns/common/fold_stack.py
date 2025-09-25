import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import get_input_kw_node, get_input_node


class FoldStack(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def _get_fold_result(self, gm: fx.GraphModule, src: fx.Node, dim: int):
        copy = gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(src,),
        )
        view = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(copy, dim))
        return view

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.stack.default
        ]

        for stack in reversed(candidates):
            inps = get_input_node(stack, 0)
            if len(inps) == 1:
                changed = True
                inp = inps[0]
                dim = get_input_kw_node(stack, "dim")
                with gm.graph.inserting_before(stack):
                    fold_res = self._get_fold_result(gm, inp, dim)
                    stack.replace_all_uses_with(fold_res)
                    gm.graph.erase_node(stack)

        return changed
