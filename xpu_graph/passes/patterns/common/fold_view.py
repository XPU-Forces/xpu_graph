import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern


class FoldView0(Pattern):
    """
    Fold aten.view which inp.shape == target_shape
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        view_tup = (
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
        )
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in view_tup]

        for view in candidates:
            inp = view.args[0]
            # Use target node's shape is more straightforward
            if list(view.meta["val"].shape) == list(inp.meta["val"].shape):
                changed = True

                view.replace_all_uses_with(inp)
                gm.graph.erase_node(view)

        return changed


_view_like_ops = (
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
)


class FoldView1(Pattern):
    """
    Fold aten.view(aten.view) -> aten.view
    """

    _support_stages = [FxStage.inference, FxStage.pregrad, FxStage.forward]

    def process(self, gm: fx.GraphModule):
        changed = False
        view_tup = (
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
        )
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in view_tup]

        for view in candidates:
            inp = view.args[0]
            if isinstance(inp, fx.Node) and inp.op == "call_function" and inp.target in _view_like_ops:
                changed = True
                view.replace_input_with(inp, inp.args[0])

        return changed
