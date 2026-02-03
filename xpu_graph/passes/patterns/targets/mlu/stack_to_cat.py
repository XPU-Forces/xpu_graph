from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_cat_op, check_stack_op, get_shape


def _is_stack_to_cat(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    stack_inputs = node.args[0]
    last_input = stack_inputs[-1]
    if len(last_input.users) == 1:
        return False, 0, None
    input_shape = get_shape(last_input)

    for key, _ in last_input.users.items():
        is_cat, axis = check_cat_op(key)
        if not is_cat:
            continue
        if key.args[0] == stack_inputs:
            return True, axis, key
    return False, 0, None


class CatToStack(Pattern):
    _opt_level = OptLevel.level1
    """
    a = stack([a, b, c, d])
    b = cat([a, b, c, d], axis = -1)
    ->
    b = a.view.trans
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        # only support 2d
        is_modified = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and check_stack_op(node) and len(get_shape(node)) == 3
        ]
        for node in candidates:
            is_change_stack, axis, cat_node = _is_stack_to_cat(node)
            if not is_change_stack:
                continue
            bs, m, n = get_shape(node)

            with graph_module.graph.inserting_after(node):
                if axis != 0:
                    transpose_node = graph_module.graph.call_function(
                        torch.ops.aten.transpose.int,
                        args=(node, 1, 0),
                        kwargs={},
                    )
                    with graph_module.graph.inserting_after(transpose_node):
                        reshape_node = graph_module.graph.call_function(
                            torch.ops.aten.reshape.default,
                            args=(transpose_node, [m, bs * n]),
                            kwargs={},
                        )
                    cat_node.replace_all_uses_with(reshape_node)
                else:
                    reshape_node = graph_module.graph.call_function(
                        torch.ops.aten.view.default,
                        args=(node, [-1, n]),
                        kwargs={},
                    )
                    cat_node.replace_all_uses_with(reshape_node)
            is_modified = True

        return is_modified
