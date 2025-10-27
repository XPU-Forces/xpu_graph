import sympy
import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

from xpu_graph.utils import logger


class SymShapeManager:
    def __init__(self, graph: torch.fx.Graph):
        r = {}
        for node in graph.nodes:
            if "val" in node.meta and isinstance(node.meta["val"], torch.SymInt):
                if node.meta["val"].node.expr not in r:
                    r[node.meta["val"].node.expr] = node
        self.sym_shape_node_map = r

    def get_shape_val(self, sym_num):
        if not isinstance(sym_num, torch.SymInt):
            return sym_num
        # Note: zero/one may be specialized
        if isinstance(sym_num.node.expr, sympy.core.numbers.Integer):
            return int(sym_num.node.expr)
        if sym_num.node.expr in self.sym_shape_node_map:
            return self.sym_shape_node_map[sym_num.node.expr]
        return None

    def rebind_shape(self, shape):
        bind_shape = [self.get_shape_val(s) for s in shape]
        if any(s is None for s in bind_shape):
            logger.debug("Cannot find binding for shape sympy expr, shape: %s, (partial result: %s)", shape, bind_shape)
            return None
        return bind_shape


def same_shape(shape, other):
    if isinstance(shape, (torch.Size, tuple)):
        shape = list(shape)
    if isinstance(other, (torch.Size, tuple)):
        other = list(other)
    return statically_known_true(sym_eq(shape, other))
