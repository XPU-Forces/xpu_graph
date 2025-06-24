from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ..utils.check_ops import check_add_op, check_mm_op, check_view, get_shape


class FusedAddN(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad, FxStage.backward]
    """
    a = add(x1, x2)
    b = add(a, x3)
    c = add(b, x4)
    -->
    stack([x1, x2, x3, x4]).sum(dim=[0])
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        add_tup = (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add.Scalar,
        )
        candidates = [
            node
            for node in reversed(graph_module.graph.nodes)
            if node.op == "call_function" and node.target in add_tup and isinstance(node.args[1], fx.Node)
        ]

        for add in candidates:
            n = add
            add_list = [n.args[1]]
            delete_list = [n]
            while True:
                inp0 = n.args[0]
                if inp0 in candidates and len(inp0.users) == 1:
                    add_list.append(inp0.args[1])
                    delete_list.append(inp0)
                    n = inp0
                else:
                    break
            add_list.append(n.args[0])
            if len(add_list) < 4:
                continue

            add_list = list(reversed(add_list))

            # assert the same shape
            shape0 = get_shape(add_list[0])
            shapes_compatible = True
            for operand in add_list[1:]:
                shape_i = get_shape(operand)
                if shape0 != shape_i:
                    shapes_compatible = False
                    break
            if not shapes_compatible:
                continue

            with graph_module.graph.inserting_before(add):
                stack_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.stack.default,
                    args=(add_list,),
                    name=add.name + "_fusedadd_stack",
                )
                sum_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.sum.dim_IntList,
                    args=(stack_node, [0]),
                    name=add.name + "_fusedadd_sum",
                )
                add.replace_all_uses_with(sum_node)
                changed = True
                for add_node in delete_list:
                    if len(add_node.users) == 0:
                        graph_module.graph.erase_node(add_node)
            if changed:
                print(graph_module.graph)
        return changed
