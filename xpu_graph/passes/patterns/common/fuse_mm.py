from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ..utils.check_ops import check_add_op, check_mm_op


def _is_mm_add(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if not check_add_op(node):
        return False, ()
    mm_node = node.args[0]
    bias_node = node.args[1]
    # Note: This pattern does not fuse residuals
    is_mm, input_node, weight_node = check_mm_op(mm_node)
    if not is_mm:
        mm_node = node.args[1]
        bias_node = node.args[0]
        is_mm, input_node, weight_node = check_mm_op(mm_node)
        if not is_mm:
            return False, ()
    if len(mm_node.users) != 1:
        return False, ()
    if isinstance(bias_node, float) or isinstance(bias_node, int) or len(bias_node.meta["val"].shape) > 2:
        return False, ()

    return True, (bias_node, input_node, weight_node)


class FusedAddMM(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        for node in reversed(graph_module.graph.nodes):
            is_mm_add, mm_inputs = _is_mm_add(node)
            if is_mm_add:
                with graph_module.graph.inserting_before(node):
                    addmm_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.addmm.default,
                        args=(mm_inputs),
                        name=node.name + "_replacement",
                    )
                node.replace_all_uses_with(addmm_node)
                graph_module.graph.erase_node(node)
                changed = True

        return changed
