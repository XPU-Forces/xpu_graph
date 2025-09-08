from typing import Dict

import torch
from torch import fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_where_op, check_zeros_op


class ConvertWhereMaskToMul(Pattern):
    """
    where(cond, zero, args) -> (~cond) * args
    where(cond, args, zero) -> cond * args
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
    ]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        cond_to_rev: Dict[fx.Node, fx.Node] = {}
        for node in graph_module.graph.nodes:
            if not check_where_op(node):
                continue
            cond, x, y = node.args[:3]
            if check_zeros_op(x):
                # mask = ~cond
                if cond not in cond_to_rev:
                    with graph_module.graph.inserting_after(cond):
                        cond_to_rev[cond] = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.logical_not.default,
                            args=(cond,),
                            name=cond.name + "_not",
                        )
                mask = cond_to_rev[cond]
                val = y
            elif check_zeros_op(y):
                mask = cond
                val = x
            else:
                continue

            # *
            with graph_module.graph.inserting_before(node):
                new_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.mul.Tensor,
                    args=(mask, val),
                    name=node.name + "_tomul",
                )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
            is_modified = True
        return is_modified
