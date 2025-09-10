from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_where_op, is_one_like, is_zero_like


def replace_where_to_mul(graph_module, nodes, cond, idx):
    nodes_to_remove = []
    for n in nodes:
        with graph_module.graph.inserting_before(n):
            target_node = n.args[idx]
            cast_node = None
            if hasattr(target_node, "meta") and "tensor_meta" in target_node.meta:
                target_dtype = target_node.meta["tensor_meta"].dtype
                if torch.bool != target_dtype:
                    cast_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten._to_copy.default,
                        args=(cond,),
                        kwargs={"dtype": target_dtype},
                        name=cond.name + "_cast",
                    )
            new_n = graph_module.graph.create_node(
                op="call_function",
                target=torch.ops.aten.mul.Tensor,
                args=(cond if cast_node == None else cast_node, target_node),
                name=n.name + "_tomul",
            )
            n.replace_all_uses_with(new_n)
            nodes_to_remove.append(n)
    for n in nodes_to_remove:
        graph_module.graph.erase_node(n)


def logical_not_cond(graph_module, cond):
    new_cond = graph_module.graph.create_node(
        op="call_function",
        target=torch.ops.aten.logical_not.default,
        args=(cond,),
        name=cond.name + "_not",
    )
    return new_cond


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
        where_cond_z_l: Dict[fx.Node, List[fx.Node]] = {}
        where_cond_z_r: Dict[fx.Node, List[fx.Node]] = {}
        for node in candidates:
            cond, x, y = node.args[:3]
            if is_zero_like(x):
                if cond not in where_cond_z_l:
                    where_cond_z_l[cond] = []
                where_cond_z_l[cond].append(node)
            elif is_zero_like(y):
                if cond not in where_cond_z_r:
                    where_cond_z_r[cond] = []
                where_cond_z_r[cond].append(node)

        for cond, nodes in where_cond_z_l.items():
            if len(nodes) < 3:
                continue
            with graph_module.graph.inserting_after(cond):
                new_cond = logical_not_cond(graph_module, cond)
                replace_where_to_mul(graph_module, nodes, new_cond, 2)
            is_modified = True

        for cond, nodes in where_cond_z_r.items():
            if len(nodes) < 2:
                continue
            replace_where_to_mul(graph_module, nodes, cond, 1)
            is_modified = True

        return is_modified
