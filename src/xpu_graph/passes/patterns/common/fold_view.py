import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.shape_utils import same_shape


class FoldView0(Pattern):
    """
    Fold aten.view which inp.shape equals target_shape
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
            torch.ops.aten.reshape.default,
        )
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in view_tup]

        for view in candidates:
            inp = view.args[0]
            # Use target node's shape is more straightforward
            if same_shape(view.meta["val"].shape, inp.meta["val"].shape):
                changed = True

                view.replace_all_uses_with(inp)
                gm.graph.erase_node(view)

        return changed


_view_like_ops = (
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
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

    def _infer_folded_shape(self, orig_shape, new_shape):
        if any(isinstance(dim, int) and dim < 0 for dim in orig_shape):
            return new_shape
        # Note: if orig_shape has no "-1", it actually performs a sym-shape assertions, and we should be careful when folding it
        if any(isinstance(dim, int) and dim < 0 for dim in new_shape):
            # The actual situation where new folded shape should be infered rather than naive folding
            def _collec_mixed_numel(shape):
                symbolic_dims = []
                concrete_dims = 1
                for dim in shape:
                    if isinstance(dim, fx.Node):
                        symbolic_dims.append(dim)
                    elif dim > 0:
                        concrete_dims *= dim
                return sorted(symbolic_dims), concrete_dims

            orig_symbolic_shapes, orig_concrete_shape = _collec_mixed_numel(orig_shape)
            new_symbolic_shapes, new_concrete_shape = _collec_mixed_numel(new_shape)
            if orig_symbolic_shapes != new_symbolic_shapes:
                if orig_concrete_shape == new_concrete_shape:
                    for orig_dim, new_dim in zip(orig_symbolic_shapes, new_symbolic_shapes + [None]):
                        if orig_dim != new_dim:
                            infered_dim = orig_dim
                            break
                else:
                    # we give up to infer the folded shape if the symbolic part is different
                    return None
            else:
                infered_dim = orig_concrete_shape // new_concrete_shape
            return tuple(infered_dim if dim == -1 else dim for dim in new_shape)
        return new_shape

    def process(self, gm: fx.GraphModule):
        changed = False
        view_tup = (
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.reshape.default,
        )
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in view_tup]

        for view in candidates:
            inp = view.args[0]
            if isinstance(inp, fx.Node) and inp.op == "call_function" and inp.target in _view_like_ops:
                if inp.target in view_tup:
                    if (folded_shape := self._infer_folded_shape(inp.args[1], view.args[1])) is not None:
                        changed = True
                        view.args = (inp.args[0], folded_shape)
                else:
                    changed = True
                    view.replace_input_with(inp, inp.args[0])

        return changed
