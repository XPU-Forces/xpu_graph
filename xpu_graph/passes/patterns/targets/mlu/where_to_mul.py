from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_where_op, check_zeros_op


class FusedWhereToMul(Pattern):
    """
    where(cond, zero, args) -> (~cond) * args
    """

    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
    ]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        candidates = [node for node in graph_module.graph.nodes if node.op == "call_function" and check_where_op(node)]
        where_cond: Dict[fx.Node, List[fx.Node]] = {}
        for node in candidates:
            cond, x, y = node.args[:3]
            if not check_zeros_op(x):
                continue
            if cond not in where_cond:
                where_cond[cond] = []
            where_cond[cond].append(node)
        for cond, nodes in where_cond.items():
            if len(nodes) < 3:
                continue
            # ~
            with graph_module.graph.inserting_after(cond):
                new_cond = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.logical_not.default,
                    args=(cond,),
                    name=cond.name + "_not",
                )
            # *
            nodes_to_remove = []
            for n in nodes:
                with graph_module.graph.inserting_before(n):
                    new_n = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.mul.Tensor,
                        args=(new_cond, n.args[2]),
                        name=n.name + "_tomul",
                    )
                    n.replace_all_uses_with(new_n)
                    nodes_to_remove.append(n)
            for n in nodes_to_remove:
                graph_module.graph.erase_node(n)
            is_modified = True
        return is_modified
