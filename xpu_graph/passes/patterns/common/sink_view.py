from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ..utils.check_ops import (
    check_act_op,
    check_add_op,
    check_div_op,
    check_mul_op,
    check_sub_op,
    check_view,
)


class SinkView(Pattern):
    _opt_level = OptLevel.level0
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        for node in reversed(graph_module.graph.nodes):
            if not check_view(node):
                continue
            if len(node.users) != 1:
                continue
            user = list(node.users)[0]
            if check_act_op(user)[0]:
                with graph_module.graph.inserting_before(user):
                    new_act = graph_module.graph.create_node(
                        op="call_function",
                        target=user.target,
                        args=(node.args[0],),
                        name=user.name + "_replacement",
                    )
                    new_act_view = graph_module.graph.create_node(
                        op="call_function",
                        target=node.target,
                        args=(new_act, node.args[1]),
                        name=node.name + "_replacement",
                    )
                user.replace_all_uses_with(new_act_view)
                graph_module.graph.erase_node(user)
                changed = True
            elif check_add_op(user) or check_sub_op(user) or check_mul_op(user) or check_div_op(user):
                if user.args[0] is node:
                    other_node = user.args[1]
                else:
                    other_node = user.args[0]
                if isinstance(other_node, float) or isinstance(other_node, int):
                    other_shape = []
                else:
                    other_shape = other_node.meta["val"].shape
                result_shape = user.meta["val"].shape
                view_shape = node.meta["val"].shape
                orig_shape = node.args[0].meta["val"].shape
                # Only if the result shape is the same as the view shape, and the bias-ed part get unchanged
                if result_shape == view_shape and (
                    len(other_shape) == 0 or orig_shape[-len(other_shape) :] == view_shape[-len(other_shape) :]
                ):
                    with graph_module.graph.inserting_before(user):
                        new_add = graph_module.graph.create_node(
                            op="call_function",
                            target=user.target,
                            args=(node.args[0], other_node),
                            name=user.name + "_replacement",
                        )
                        new_add_view = graph_module.graph.create_node(
                            op="call_function",
                            target=node.target,
                            args=(new_add, node.args[1]),
                            name=node.name + "_replacement",
                        )
                    user.replace_all_uses_with(new_add_view)
                    graph_module.graph.erase_node(user)
                    changed = True

        return changed
