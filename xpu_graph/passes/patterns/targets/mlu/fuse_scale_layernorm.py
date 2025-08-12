import operator
from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_getitem_op, check_mul_op, check_norm_module


def is_scale_layernorm(
    layernorm_node: fx.Node,
) -> bool:
    is_norm, mode = check_norm_module(layernorm_node)
    if not is_norm or mode != "layer_norm":
        return False
    scale_node, weight, bias, eps = layernorm_node.args
    if len(scale_node.users) != 1:
        return False
    if not check_mul_op(scale_node):
        return False
    scale = scale_node.args[1]
    if not isinstance(scale, (int, float)):
        return False

    eps = layernorm_node.args[3]
    if not isinstance(eps, (int, float)):
        return False
    if scale == 0:
        return False
    new_eps = eps / scale / scale
    layernorm_node.args = (scale_node.args[0], weight, bias, new_eps)
    return True


class FusedScaleLayernorm(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        for node in reversed(graph_module.graph.nodes):
            is_modified |= is_scale_layernorm(node)

        return is_modified
