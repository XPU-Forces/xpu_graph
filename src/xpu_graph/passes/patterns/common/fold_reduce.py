from typing import List

import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import get_input_kw_node, get_input_node
from xpu_graph.passes.patterns.utils.shape_utils import same_shape

reduce_tup = (torch.ops.aten.sum.dim_IntList,)


class FoldReduce(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def _get_fold_result(self, gm: fx.GraphModule, src, dims: List[int], keep_dim: bool) -> fx.Node:
        copy = gm.graph.call_function(
            torch.ops.aten.clone.default,
            args=(src,),
            kwargs={
                "memory_format": torch.contiguous_format,
            },
        )
        if keep_dim:
            return copy
        else:
            view = gm.graph.call_function(
                torch.ops.aten.squeeze.dims,
                args=(
                    copy,
                    dims,
                ),
            )
            return view

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in reduce_tup]

        for reduce in reversed(candidates):
            inp = get_input_node(reduce, 0)
            shape = inp.meta["val"].shape
            dims = get_input_kw_node(reduce, "dim") or list(range(len(shape)))
            if not isinstance(dims, list):
                dims = [dims]
            keep_dim = get_input_kw_node(reduce, "keepdim") or False
            if all(same_shape(shape[dim], 1) for dim in dims):
                changed = True
                with gm.graph.inserting_before(reduce):
                    fold_res = self._get_fold_result(gm, inp, dims, keep_dim)
                    reduce.replace_all_uses_with(fold_res)
                    gm.graph.erase_node(reduce)

        return changed
