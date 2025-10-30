from typing import Optional, Tuple, Union

import torch

aten = torch.ops.aten
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DefaultLayerNorm
from xpu_graph.passes.patterns.utils.shape_utils import same_shape

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


def _is_unbiased_layernorm(node: fx.Node, maybe_layernorm_nodes=None):
    if check_mul_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_unaffined(node):
            if maybe_layernorm_nodes is not None:
                return (
                    node in maybe_layernorm_nodes
                    and maybe_layernorm_nodes[node][1] is None
                    and maybe_layernorm_nodes[node][2] is None
                )
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

        return True, [unaffined, weight]

    return False, None


def _is_layernorm(node: fx.Node, maybe_layernorm_nodes=None):
    if check_add_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_unbiased(node):
            if maybe_layernorm_nodes is not None:
                return node in maybe_layernorm_nodes and maybe_layernorm_nodes[node][2] is None
            if not isinstance(node, fx.Node) or node.op != "call_module":
                return False
            target_mod = getattr(node.graph.owning_module, node.target)
            return isinstance(target_mod, DefaultLayerNorm) and node.args[2] is None

        def _is_casted_unbiased(node):
            if not is_type_cast(node):
                return False
            inner = get_input_node(node, 0)
            return _is_unbiased(inner)

        if _is_unbiased(arg0):
            unbiased, bias = arg0, arg1
        elif _is_unbiased(arg1):
            unbiased, bias = arg1, arg0
        elif _is_casted_unbiased(arg0):
            unbiased = get_input_node(arg0, 0)
            bias = arg1
        elif _is_casted_unbiased(arg1):
            unbiased = get_input_node(arg1, 0)
            bias = arg0
        else:
            return False, None

        return True, [unbiased, bias]

    return False, None


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
                if not same_shape(get_shape(inputs)[-1:], get_shape(weight)):
                    continue

                with graph_module.graph.inserting_before(node):
                    layer_norm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, None, eps))
                    if inputs.meta["val"].dtype != node.meta["val"].dtype:
                        layer_norm_node = graph_module.graph.call_function(
                            aten.to.dtype if self._current_stage is FxStage.pregrad else aten._to_copy.default,
                            (layer_norm_node,),
                            {"dtype": node.meta["val"].dtype},
                        )
                node.replace_all_uses_with(layer_norm_node, propagate_meta=True)
                changed = True
            elif check_add_op(node):
                matched, params = _is_layernorm(node)
                if not matched:
                    continue
                unbiased, bias = params
                inputs, weight, _, eps = unbiased.args
                if not same_shape(get_shape(inputs)[-1:], get_shape(bias)):
                    continue

                with graph_module.graph.inserting_before(node):
                    layer_norm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, bias, eps))
                    if inputs.meta["val"].dtype != node.meta["val"].dtype:
                        layer_norm_node = graph_module.graph.call_function(
                            aten.to.dtype if self._current_stage is FxStage.pregrad else aten._to_copy.default,
                            (layer_norm_node,),
                            {"dtype": node.meta["val"].dtype},
                        )
                node.replace_all_uses_with(layer_norm_node, propagate_meta=True)
                changed = True

        return changed


from torch._inductor.pattern_matcher import joint_fwd_bwd
from torch.fx.experimental.optimization import extract_subgraph
from torch.fx.subgraph_rewriter import replace_pattern


class FusedLayerNormJoint(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.joint]

    def process(self, graph_module):
        is_modified = False
        _maybe_layernorm_nodes = {}
        for node in graph_module.graph.nodes:
            # print(node.format_node())
            # breakpoint()
            if node.meta.get("partitioner_tag", None) == "is_backward":
                continue
            matched, params = _is_unaffined_layernorm(node)
            if matched:
                input, eps = params
                _maybe_layernorm_nodes[node] = (input, None, None, eps)
                continue
            matched, params = _is_unbiased_layernorm(node, _maybe_layernorm_nodes)
            if matched:
                unaffined, weight = params
                input, _, _, eps = _maybe_layernorm_nodes.pop(unaffined)
                _maybe_layernorm_nodes[node] = (input, weight, None, eps)
                continue
            matched, params = _is_layernorm(node, _maybe_layernorm_nodes)
            if matched:
                unbiased, bias = params
                input, weight, _, eps = _maybe_layernorm_nodes.pop(unbiased)
                _maybe_layernorm_nodes[node] = (input, weight, bias, eps)
                continue
        print(_maybe_layernorm_nodes)

        def _extract_medium_nodes(maybe_layernorm_result, maybe_layernorm_args):
            mediums = []
            explored_nodes = [maybe_layernorm_result]
            while len(explored_nodes) > 0:
                cur_node = explored_nodes.pop(0)
                for prev_node in cur_node.args:
                    if not isinstance(prev_node, fx.Node):
                        continue
                    if prev_node in maybe_layernorm_args or prev_node in mediums:
                        continue
                    mediums.append(prev_node)
                    explored_nodes.append(prev_node)
            return mediums

        for layernorm_node, params in _maybe_layernorm_nodes.items():
            input, weight, bias, eps = params
            mediums = _extract_medium_nodes(layernorm_node, [input, weight, bias])
            fw_inp_nodes = [p for p in params if isinstance(p, fx.Node)]
            fw_pat = extract_subgraph(
                graph_module,
                nodes=sorted(mediums) + [layernorm_node],
                inputs=fw_inp_nodes,
                outputs=[layernorm_node],
            )
            fw_inps = [inp_node.meta["val"].detach().requires_grad_() for inp_node in fw_inp_nodes]
            # fw_inps[0].requires_grad = False
            joint_pat = joint_fwd_bwd(fw_pat, args=fw_inps)

            def target_fn(inp, weight, bias):
                return torch.nn.functional.layer_norm(inp, inp.shape[-1:], weight, bias, eps)

            joint_target = joint_fwd_bwd(target_fn, args=fw_inps)
            joint_pat.print_readable()
            joint_target.print_readable()
            matched = replace_pattern(graph_module, joint_pat, joint_target)
            if len(matched) > 0:
                is_modified = True
        return is_modified


class RemoveLayerNormCast(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_layer_norm"):
            graph_module.add_module("fused_layer_norm", DefaultLayerNorm())
        for node in reversed(graph_module.graph.nodes):
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), DefaultLayerNorm):
                # since all internal operations should have been promoted to f32, its okay to remove all typecasts
                # and it will do no harm to throughput
                do_replace = False
                inputs, weight, bias, eps = node.args
                if is_type_cast(inputs):
                    inputs = get_input_node(inputs, 0)
                    do_replace = True
                if is_type_cast(weight):
                    weight = get_input_node(weight, 0)
                    do_replace = True
                if is_type_cast(bias):
                    bias = get_input_node(bias, 0)
                    do_replace = True
                if do_replace:
                    if len(node.users) == 1 and is_type_cast(list(node.users)[0]):
                        result_node = list(node.users)[0]
                    else:
                        result_node = node
                    with graph_module.graph.inserting_before(node):
                        new_layernorm = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, bias, eps))

                        if inputs.meta["val"].dtype != result_node.meta["val"].dtype:
                            new_layernorm = graph_module.graph.call_function(
                                aten.to.dtype if self._current_stage is FxStage.pregrad else aten._to_copy.default,
                                (new_layernorm,),
                                {"dtype": result_node.meta["val"].dtype},
                            )
                        result_node.replace_all_uses_with(new_layernorm, propagate_meta=True)
                        is_modified = True

        return is_modified
