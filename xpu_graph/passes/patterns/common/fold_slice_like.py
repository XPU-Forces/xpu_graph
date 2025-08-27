import torch
import torch.fx as fx
from torch import SymInt

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

MAX_INT64 = 9223372036854775807


class FoldSliceLike(Pattern):
    """
    Folds redundant slice-like operations, including `slice` and `slice_scatter`.

    1. FoldSlice:
       If a `slice` operation covers the entire dimension of the input tensor,
       it's a no-op and can be replaced by the input tensor itself.
       Pattern: y = slice(x, dim, 0, len(x)(or inf)) -> Becomes: y = x

    2. FoldSliceScatter:
       If a `slice_scatter` operation writes a `view` tensor into a `base` tensor
       and the target slice covers the entire `base` tensor, the operation is
       equivalent to just returning the `view`.
       Pattern: y = slice_scatter(base, view, ...) -> Becomes: y = view
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue

            if node.target == torch.ops.aten.slice.Tensor:
                if self._fold_slice(node, gm):
                    logger.info(f"FoldSliceLike: Folded slice node {node.name}")
                    changed = True

            elif node.target == torch.ops.aten.slice_scatter.default:
                if self._fold_slice_scatter(node, gm):
                    logger.info(f"FoldSliceLike: Folded slice_scatter node {node.name}")
                    changed = True

        return changed

    def _fold_slice(self, node: fx.Node, gm: fx.GraphModule) -> bool:
        src_node, dim, start, end = node.args[0], node.args[1], node.args[2], node.args[3]

        if "val" not in src_node.meta:
            return False
        src_shape = src_node.meta["val"].shape
        if dim >= len(src_shape):
            return False

        dim_length = src_shape[dim]

        is_noop = False
        if start == 0 and end >= MAX_INT64:
            is_noop = True
        if start == 0 and end == dim_length:
            is_noop = True

        if is_noop:
            node.replace_all_uses_with(src_node)
            gm.graph.erase_node(node)
            return True
        return False

    def _fold_slice_scatter(self, node: fx.Node, gm: fx.GraphModule) -> bool:
        if len(node.args) < 5:
            return False
        base_node, view_node, dim, start, end = node.args[:5]

        if not all(isinstance(n, fx.Node) and "val" in n.meta for n in [base_node, view_node]):
            return False
        if not isinstance(dim, int) or not isinstance(start, int):
            return False

        base_shape = base_node.meta["val"].shape
        view_shape = view_node.meta["val"].shape

        if base_shape != view_shape:
            return False

        if start != 0:
            return False

        dim_length = view_shape[dim]
        if end != dim_length:
            return False

        node.replace_all_uses_with(view_node)
        gm.graph.erase_node(node)
        return True
