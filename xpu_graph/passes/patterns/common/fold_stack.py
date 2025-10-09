import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import (
    find_common_src,
    get_input_kw_node,
    get_input_node,
)


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

    def _get_fold_unbind_stack_result(self, gm: fx.GraphModule, unbind_node: fx.Node, stack_node: fx.Node):
        unbind_dim = 0
        if len(unbind_node.args) == 2:
            unbind_dim = unbind_node.args[1]
        elif "dim" in unbind_node.kwargs:
            unbind_dim = unbind_node.kwargs["dim"]
        if unbind_dim < 0:
            unbind_dim += len(unbind_node.args[0].meta["val"].shape)
        stack_dim = 0
        if len(stack_node.args) == 2:
            stack_dim = stack_node.args[1]
        elif "dim" in stack_node.kwargs:
            stack_dim = stack_node.kwargs["dim"]
        if stack_dim < 0:
            stack_dim += len(stack_node.meta["val"].shape)

        if unbind_dim == stack_dim:
            return gm.graph.call_function(
                torch.ops.aten.clone.default,
                args=(unbind_node.args[0],),
            )
        else:
            return None

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
            else:
                unbind_src = find_common_src(inps, torch.ops.aten.unbind.int)
                if unbind_src is not None:
                    with gm.graph.inserting_before(stack):
                        fold_res = self._get_fold_unbind_stack_result(gm, unbind_src, stack)
                        if fold_res is not None:
                            stack.replace_all_uses_with(fold_res)
                            gm.graph.erase_node(stack)
                            changed = True

        return changed
