from typing import Optional, Union, Tuple

import torch
from torch import nn, fx
from typing import Callable, Optional, List
from xpu_graph.config import OptLevel

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ..utils.check_ops import (
    check_add_op,
    check_sub_op,
    check_mul_op,
    check_div_or_mul_op,
    check_pow_op,
    check_mean_op,
    check_var_op,
    check_sqrt_op,
    check_rsqrt_op,
    get_input_node,
    get_actual_node,
    get_input_kw_node,
    get_shape,
)


def _is_unaffined_layernorm(
    node: fx.Node,
) -> Tuple[bool, Optional[Tuple[fx.Node, Optional[Union[float, int]]]]]:
    # Matching: y = (x - mean(x)) / sqrt(var(x) + eps)
    # Or:       y = (x - mean(x)) * rsqrt(var(x) + eps)
    matched, node0, node1 = check_div_or_mul_op(node)
    if not matched:
        return False, None
    sub = node0
    if not check_sub_op(sub):
        return False, None
    input = get_actual_node(node0, 0)
    # if len(get_shape(input)) <= 2:
    #     return False, None
    mean = get_input_node(node0, 1)
    if (
        not check_mean_op(mean)
        or get_actual_node(mean, 0) != input
        or get_input_node(mean, 1) != [-1]
        or get_input_node(mean, 2) != True
    ):
        return False, None

    sqrt, is_div = node1
    if is_div:
        if check_sqrt_op(sqrt) or (
            check_pow_op(sqrt) and get_input_node(sqrt, 1) == 0.5
        ):
            plus = get_input_node(sqrt, 0)
        else:
            return False, None
    else:
        if check_rsqrt_op(sqrt) or (
            check_pow_op(sqrt) and get_input_node(sqrt, 1) == -0.5
        ):
            plus = get_input_node(sqrt, 0)
        else:
            return False, None

    if not check_add_op(plus):
        var = plus
        eps = None
        if not check_var_op(var):
            return False, None
    else:
        var = get_input_node(plus, 0)
        eps = get_input_node(plus, 1)
        if not isinstance(eps, (float, int)):
            var, eps = eps, var

    if (
        get_actual_node(var, 0) != input
        or get_input_node(var, 1) != [-1]
        or get_input_kw_node(var, "keepdim") != True
        or not isinstance(eps, (float, int))
        or (
            get_input_kw_node(var, "unbiased") != False
            and get_input_kw_node(var, "correction") != 0
        )
    ):
        return False, None
    return True, (input, eps)


def _is_unbiased_layernorm(
    node: fx.Node,
) -> Tuple[bool, Optional[Tuple[fx.Node, Optional[Union[float, int]], Optional[fx.Node]]]]:
    res, nodes = _is_unaffined_layernorm(node)
    if res:
        input, eps = nodes
        return True, (input, eps, None)

    if not check_mul_op(node):
        return False, None

    unaffined_ln = get_input_node(node, 0)
    weight = get_input_node(node, 1)
    res, nodes = _is_unaffined_layernorm(unaffined_ln)
    if res:
        input, eps = nodes
        if get_shape(input)[-1:] != get_shape(weight):
            return False, None
        return True, (input, eps, weight)

    weight = get_input_node(node, 0)
    unaffined_ln = get_input_node(node, 1)
    res, nodes = _is_unaffined_layernorm(unaffined_ln)
    if res:
        input, eps = nodes
        if get_shape(input)[-1:] != get_shape(weight):
            return False, None
        return True, (input, eps, weight)

    return False, None


def _is_layernorm(
    node: fx.Node,
) -> Tuple[
    bool,
    Optional[
        Tuple[
            fx.Node,
            Optional[Union[float, int]],
            Optional[fx.Node],
            Optional[fx.Node],
            list[fx.Node],
        ]
    ],
]:
    res, nodes = _is_unbiased_layernorm(node)
    if res:
        input, eps, weight = nodes
        return True, (input, eps, weight, None)

    if not check_add_op(node):
        return False, None

    unbiased_ln = get_input_node(node, 0)
    bias = get_input_node(node, 1)
    res, nodes = _is_unbiased_layernorm(unbiased_ln)
    if res:
        input, eps, weight = nodes
        if get_shape(input)[-1:] != get_shape(bias):
            return False, None
        return True, (input, eps, weight, bias)

    bias = get_input_node(node, 0)
    unbiased_ln = get_input_node(node, 1)
    res, nodes = _is_unbiased_layernorm(unbiased_ln)
    if res:
        input, eps, weight = nodes
        if get_shape(input)[-1:] != get_shape(bias):
            return False, None
        return True, (input, eps, weight, bias)

    return False, None


class FusedLayerNorm(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod: torch.nn.Module):
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:

        changed = False
        graph_module.add_submodule("layer_norm_op", self.target_mod())

        for node in reversed(graph_module.graph.nodes):
            # Note: This pattern does not fuse residuals
            res, layer_norm_params = _is_layernorm(node)
            if not res:
                continue
            input, eps, weight, bias = layer_norm_params

            if eps is None:
                eps = 1e-6

            with graph_module.graph.inserting_before(node):
                layer_norm_node = graph_module.graph.call_module(
                    "layer_norm_op", args=(input, weight, bias, eps)
                )

            node.replace_all_uses_with(layer_norm_node)
            changed = True

        graph_module.graph.lint()
        graph_module.recompile()
        return changed
