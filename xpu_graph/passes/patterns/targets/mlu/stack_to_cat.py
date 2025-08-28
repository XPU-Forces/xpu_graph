from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_stack_op, check_cat_op, get_shape


def _is_stack_to_cat(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    # only support 2d
    is_cat, cat_dim = check_cat_op(node)

    cat_inputs = node.args[0]
    last_input = cat_inputs[-1]
    if len(last_input.users) == 1:
        return False, ()
    input_shape = get_shape(last_input)

    for key, _ in last_input.users.items():
        if check_stack_op(key) and key.args[0] == cat_inputs:
            args_len = len(key.args)
            if args_len == 1 or (len(key.args) == 2 or key.args[1] == 0):
                input_nums = len(cat_inputs)
                return True, (key, input_shape, input_nums)
    return False, ()


class StackToCat(Pattern):
    _opt_level = OptLevel.level1
    """
    a = stack([a, b, c, d])
    b = cat([a, b, c, d], axis = -1)
    ->
    b = a.view.trans
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.cat.default
            and len(get_shape(node)) == 2
        ]
        for node in candidates:
            is_change_stack, stack_params = _is_stack_to_cat(node)
            if not is_change_stack:
                continue

            axis = 0
            if len(node.args) == 2:
                axis = node.args[1]

            with graph_module.graph.inserting_before(stack_params[0]):
                if axis != 0:
                    reshape_node = graph_module.graph.call_function(
                        torch.ops.aten.view.default,
                        args=(
                            node,
                            [
                                stack_params[1][0],
                                stack_params[2],
                                stack_params[1][1],
                            ],
                        ),
                        kwargs={},
                    )
                    transpose_node = graph_module.graph.call_function(
                        torch.ops.aten.transpose.int,
                        args=(reshape_node, 1, 0),
                        kwargs={},
                    )
                    stack_params[0].replace_all_uses_with(transpose_node)
                else:
                    reshape_node = graph_module.graph.call_function(
                        torch.ops.aten.view.default,
                        args=(
                            node,
                            [
                                stack_params[2],
                                stack_params[1][0],
                                stack_params[1][1],
                            ],
                        ),
                        kwargs={},
                    )
                    stack_params[0].replace_all_uses_with(reshape_node)
            graph_module.graph.erase_node(stack_params[0])
            is_modified = True

        return is_modified
