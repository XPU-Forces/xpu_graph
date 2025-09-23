import sympy
import torch

from xpu_graph.utils import logger


def get_sym_shape_bindings(
    graph: torch.fx.Graph,
) -> dict[sympy.Expr, torch.fx.Node]:
    r = {}
    # NB: Prefer first occurrence of symbol
    for node in graph.nodes:
        if "val" in node.meta and isinstance(node.meta["val"], torch.SymInt):
            if node.meta["val"].node.expr not in r:
                r[node.meta["val"].node.expr] = node

    return r


def get_shape_val(sym_num, sym_shape_node_map):
    if not isinstance(sym_num, torch.SymInt):
        return sym_num
    # Note: zero/one is specialized
    if sym_num == 1:
        return 1
    if sym_num == 0:
        return 0
    if sym_num.node.expr in sym_shape_node_map:
        return sym_shape_node_map[sym_num.node.expr]
    return sym_num


def rebind_shape(shape, sym_shape_node_map):
    bind_shape = [get_shape_val(s, sym_shape_node_map) for s in shape]
    if any(isinstance(s, torch.SymInt) for s in bind_shape):
        logger.debug("Cannot find binding for shape sympy expr, shape: %s, (partial result: %s)", shape, bind_shape)
        return None
    return bind_shape
