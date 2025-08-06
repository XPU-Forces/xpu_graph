import torch

aten = torch.ops.aten
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DefaultRMSNorm

from ..utils.check_ops import (
    check_add_op,
    check_div_or_mul_op,
    check_mean_op,
    check_mul_op,
    check_pow_op,
    check_rsqrt_op,
    check_sqrt_op,
    check_square_op,
    get_input_kw_node,
    get_input_node,
    get_shape,
    is_exclusively_used,
    is_type_cast,
)


def _is_unaffined_rmsnorm(node: fx.Node):
    matched, inputs, rms = check_div_or_mul_op(node)
    if not matched:
        return False, None

    rms_node, is_div = rms
    if is_div:
        if check_sqrt_op(rms_node) or (check_pow_op(rms_node) and get_input_node(rms_node, 1) == 0.5):
            eps_mean_square = get_input_node(rms_node, 0)
        else:
            return False, None
    else:
        if check_rsqrt_op(rms_node) or (check_pow_op(rms_node) and get_input_node(rms_node, 1) == -0.5):
            eps_mean_square = get_input_node(rms_node, 0)
        elif check_rsqrt_op(inputs) or (check_pow_op(inputs) and get_input_node(inputs, 1) == -0.5):
            inputs, rms_node = rms_node, inputs
            eps_mean_square = get_input_node(rms_node, 0)
        else:
            return False, None

    if not is_exclusively_used(rms_node, node) or not is_exclusively_used(eps_mean_square, rms_node):
        return False, None

    if check_add_op(eps_mean_square):
        mean_square = get_input_node(eps_mean_square, 0)
        eps = get_input_node(eps_mean_square, 1)
        if not isinstance(eps, (float, int)):
            mean_square, eps = eps, mean_square
            if not isinstance(eps, (float, int)):
                return False, None
    else:
        return False, None

    if not is_exclusively_used(mean_square, eps_mean_square):
        return False, None

    if (
        not check_mean_op(mean_square)
        or get_input_node(mean_square, 1) != [-1]
        or get_input_kw_node(mean_square, "keepdim") != True
    ):
        return False, None

    square = get_input_node(mean_square, 0)
    if not is_exclusively_used(square, mean_square):
        return False, None

    if check_pow_op(square):
        if get_input_node(square, 0) is not inputs or get_input_node(square, 1) != 2:
            return False, None
    elif check_mul_op(square):
        if get_input_node(square, 0) is not inputs or get_input_node(square, 1) is not inputs:
            return False, None
    elif check_square_op(square):
        if get_input_node(square, 0) is not inputs:
            return False, None
    else:
        return False, None

    return True, (inputs, eps)


def _is_rmsnorm(node: fx.Node):
    if check_mul_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_unaffined(node):
            if not isinstance(node, fx.Node) or node.op != "call_module":
                return False
            target_mod = getattr(node.graph.owning_module, node.target)
            return isinstance(target_mod, DefaultRMSNorm) and node.args[1] is None

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


class FusedRMSNorm(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_rms_norm"):
            graph_module.add_module("fused_rms_norm", DefaultRMSNorm())
        for node in graph_module.graph.nodes:
            matched, rms_norm_param = _is_unaffined_rmsnorm(node)
            if matched:
                inputs, eps = rms_norm_param

                with graph_module.graph.inserting_before(node):
                    rms_norm_node = graph_module.graph.call_module("fused_rms_norm", (inputs, None, eps))
                    if inputs.meta["val"].dtype != node.meta["val"].dtype:
                        rms_norm_node = graph_module.graph.call_function(
                            aten.to.default if self._current_stage is FxStage.pregrad else aten._to_copy.default,
                            (rms_norm_node,),
                            {"dtype": node.meta["val"].dtype},
                        )
                node.replace_all_uses_with(rms_norm_node, propagate_meta=True)
                is_modified = True
            elif check_mul_op(node):
                res, rms_norm_param = _is_rmsnorm(node)
                if not res:
                    continue
                unaffined, weight = rms_norm_param
                inputs, _, eps = unaffined.args
                with graph_module.graph.inserting_before(node):
                    rms_norm_node = graph_module.graph.call_module("fused_rms_norm", (inputs, weight, eps))
                    if inputs.meta["val"].dtype != node.meta["val"].dtype:
                        rms_norm_node = graph_module.graph.call_function(
                            aten.to.default if self._current_stage is FxStage.pregrad else aten._to_copy.default,
                            (rms_norm_node,),
                            {"dtype": node.meta["val"].dtype},
                        )
                node.replace_all_uses_with(rms_norm_node, propagate_meta=True)
                is_modified = True

        return is_modified


class RemoveRMSNormCast(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_rms_norm"):
            graph_module.add_module("fused_rms_norm", DefaultRMSNorm())
        for node in reversed(graph_module.graph.nodes):
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), DefaultRMSNorm):
                inputs, weight, eps = node.args
                do_replace = False
                if is_type_cast(inputs):
                    do_replace = True
                    inputs = get_input_node(inputs, 0)
                if is_type_cast(weight):
                    do_replace = True
                    weight = get_input_node(weight, 0)
                if do_replace:
                    if len(node.users) == 1 and is_type_cast(list(node.users)[0]):
                        result_node = list(node.users)[0]
                    else:
                        result_node = node
                    with graph_module.graph.inserting_before(node):
                        new_rmsnorm = graph_module.graph.call_module("fused_rms_norm", (inputs, weight, eps))
                        if inputs.meta["val"].dtype != result_node.meta["val"].dtype:
                            new_rmsnorm = graph_module.graph.call_function(
                                (aten.to.default if self._current_stage is FxStage.pregrad else aten._to_copy.default),
                                (new_rmsnorm,),
                                {"dtype": result_node.meta["val"].dtype},
                            )
                        result_node.replace_all_uses_with(new_rmsnorm, replace_meta=True)
                        is_modified = True
        return is_modified
