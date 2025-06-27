from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_mul_op, check_norm_op


def is_scale_layernorm(
    layernorm_node: fx.Node,
) -> bool:
    is_norm, mode = check_norm_op(layernorm_node)
    if not is_norm or mode != "layer_norm":
        return False
    scale_node = layernorm_node.args[0]
    if len(scale_node.users) != 1:
        return False
    if not check_mul_op(scale_node):
        return False
    scale = scale_node.args[1]
    if not isinstance(scale, (int, float)):
        return False

    eps = layernorm_node.args[4]
    if not isinstance(eps, (int, float)):
        return False
    if scale == 0:
        return False
    args = list(layernorm_node.args)
    args[4] = eps / scale / scale
    args[0] = scale_node.args[0]
    layernorm_node.args = tuple(args)
    return True


class FusedScaleLayernorm(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        for node in reversed(graph_module.graph.nodes):
            is_modified |= is_scale_layernorm(node)

        return is_modified
