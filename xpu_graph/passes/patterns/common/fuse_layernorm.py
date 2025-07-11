from typing import Optional, Tuple, Union

from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DefaultLayerNorm

from ..utils.check_ops import (
    check_add_op,
    check_div_or_mul_op,
    check_mean_op,
    check_mul_op,
    check_pow_op,
    check_rsqrt_op,
    check_sqrt_op,
    check_sub_op,
    check_var_op,
    get_input_kw_node,
    get_input_node,
    get_shape,
    is_exclusively_used,
    is_type_cast,
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
    input = get_input_node(node0, 0)
    # if len(get_shape(input)) <= 2:
    #     return False, None
    mean = get_input_node(node0, 1)
    if (
        not check_mean_op(mean)
        or get_input_node(mean, 0) != input
        or get_input_node(mean, 1) != [-1]
        or get_input_node(mean, 2) != True
    ):
        return False, None
    if not is_exclusively_used(mean, node0):
        return False, None

    sqrt, is_div = node1
    if is_div:
        if check_sqrt_op(sqrt) or (check_pow_op(sqrt) and get_input_node(sqrt, 1) == 0.5):
            plus = get_input_node(sqrt, 0)
        else:
            return False, None
    else:
        if check_rsqrt_op(sqrt) or (check_pow_op(sqrt) and get_input_node(sqrt, 1) == -0.5):
            plus = get_input_node(sqrt, 0)
        else:
            return False, None
    if not is_exclusively_used(plus, sqrt):
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
    if not is_exclusively_used(var, plus):
        return False, None

    if (
        get_input_node(var, 0) != input
        or get_input_node(var, 1) != [-1]
        or get_input_kw_node(var, "keepdim") != True
        or not isinstance(eps, (float, int))
        or (get_input_kw_node(var, "unbiased") != False and get_input_kw_node(var, "correction") != 0)
    ):
        return False, None
    return True, (input, eps)


def _is_unbiased_layernorm(node: fx.Node):
    if check_mul_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_unaffined(node):
            if not isinstance(node, fx.Node) or node.op != "call_module":
                return False
            target_mod = getattr(node.graph.owning_module, node.target)
            return isinstance(target_mod, DefaultLayerNorm) and node.args[1] is None and node.args[2] is None

        if _is_unaffined(arg0):
            unaffined, weight = arg0, arg1
        elif _is_unaffined(arg1):
            unaffined, weight = arg1, arg0
        else:
            return False, None

        if get_shape(unaffined)[-1:] != get_shape(weight):
            return False, None
        else:
            return True, [unaffined, weight]

    return False, None


def _is_layernorm(node: fx.Node):
    if check_add_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_unbiased(node):
            if not isinstance(node, fx.Node) or node.op != "call_module":
                return False
            target_mod = getattr(node.graph.owning_module, node.target)
            return isinstance(target_mod, DefaultLayerNorm) and node.args[2] is None

        if _is_unbiased(arg0):
            unbiased, bias = arg0, arg1
        elif _is_unbiased(arg1):
            unbiased, bias = arg1, arg0
        else:
            return False, None

        if get_shape(unbiased)[-1:] != get_shape(bias):
            return False, None
        else:
            return True, [unbiased, bias]

    return False, None


def _is_casted_layernorm(node: fx.Node):
    if not is_type_cast(node):
        return False
    inner = get_input_node(node, 0)
    if not is_exclusively_used(inner, node):
        return False
    if inner.op == "call_module" and isinstance(getattr(inner.graph.owning_module, inner.target), DefaultLayerNorm):
        inputs = inner.args[0]
        if not is_type_cast(inputs):
            return False
        real_inputs = get_input_node(inputs, 0)
        return real_inputs.meta["val"].dtype == node.meta["val"].dtype

    return False


class FusedLayerNorm(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        if not hasattr(graph_module, "fused_layer_norm"):
            graph_module.add_module("fused_layer_norm", DefaultLayerNorm())

        for node in graph_module.graph.nodes:
            # Note: This pattern does not fuse residuals
            matched, params = _is_unaffined_layernorm(node)
            if matched:
                input, eps = params

                with graph_module.graph.inserting_before(node):
                    layer_norm_node = graph_module.graph.call_module("fused_layer_norm", (input, None, None, eps))

                node.replace_all_uses_with(layer_norm_node, propagate_meta=True)
                changed = True
            elif check_mul_op(node):
                matched, params = _is_unbiased_layernorm(node)
                if not matched:
                    continue
                unaffined, weight = params
                inputs, _, _, eps = unaffined.args

                with graph_module.graph.inserting_before(node):
                    layer_norm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, None, eps))

                node.replace_all_uses_with(layer_norm_node, propagate_meta=True)
                changed = True
            elif check_add_op(node):
                matched, params = _is_layernorm(node)
                if not matched:
                    continue
                unbiased, bias = params
                inputs, weight, _, eps = unbiased.args

                with graph_module.graph.inserting_before(node):
                    layer_norm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, bias, eps))

                node.replace_all_uses_with(layer_norm_node, propagate_meta=True)
                changed = True

        return changed


class RemoveLayerNormCast(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_layer_norm"):
            graph_module.add_module("fused_layer_norm", DefaultLayerNorm())
        for node in reversed(graph_module.graph.nodes):
            if _is_casted_layernorm(node):
                layer_norm_node = get_input_node(node, 0)
                inputs, weight, bias, eps = layer_norm_node.args
                real_inputs = get_input_node(inputs, 0)
                with graph_module.graph.inserting_before(node):
                    new_rmsnorm = graph_module.graph.call_module("fused_layer_norm", (real_inputs, weight, bias, eps))

                node.replace_all_uses_with(new_rmsnorm)
                is_modified = True
        return is_modified
