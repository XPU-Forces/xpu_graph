import torch
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
    is_exclusively_used,
    is_type_cast,
)


def _is_unaffined_layernorm(node: fx.Node):
    # Pattern: (inputs - mean(inputs)) / sqrt(var(inputs) + eps)
    # From back to front matching:
    # node: div_or_mul_op
    # ├── sub_op (inputs - mean)
    # │   ├── inputs
    # │   └── mean_op
    # └── sqrt_op/rsqrt_op
    #     └── add_op (var + eps)
    #         ├── var_op
    #         └── eps (float/int)
    
    matched, sub_node, sqrt_rhs = check_div_or_mul_op(node)
    if not matched:
        return False, None

    sqrt_node, is_div = sqrt_rhs
    
    # Check subtraction: (inputs - mean)
    if not check_sub_op(sub_node):
        return False, None
    
    inputs = get_input_node(sub_node, 0)
    mean_node = get_input_node(sub_node, 1)
    
    if not is_exclusively_used(sub_node, node):
        return False, None

    # Check mean calculation
    if (
        not check_mean_op(mean_node)
        or get_input_node(mean_node, 0) != inputs
        or get_input_node(mean_node, 1) != [-1]
        or get_input_kw_node(mean_node, "keepdim") != True
    ):
        return False, None

    if not is_exclusively_used(mean_node, sub_node):
        return False, None

    # Check sqrt/rsqrt operation
    if is_div:
        # Division case: / sqrt(var + eps)
        if check_sqrt_op(sqrt_node) or (check_pow_op(sqrt_node) and get_input_node(sqrt_node, 1) == 0.5):
            var_eps_node = get_input_node(sqrt_node, 0)
        else:
            return False, None
    else:
        # Multiplication case: * rsqrt(var + eps)
        if check_rsqrt_op(sqrt_node) or (check_pow_op(sqrt_node) and get_input_node(sqrt_node, 1) == -0.5):
            var_eps_node = get_input_node(sqrt_node, 0)
        else:
            return False, None

    if not is_exclusively_used(sqrt_node, node) or not is_exclusively_used(var_eps_node, sqrt_node):
        return False, None

    # Check variance + eps
    if check_add_op(var_eps_node):
        var_node = get_input_node(var_eps_node, 0)
        eps = get_input_node(var_eps_node, 1)
        if not isinstance(eps, (float, int)):
            var_node, eps = eps, var_node
            if not isinstance(eps, (float, int)):
                return False, None
    else:
        var_node = var_eps_node
        eps = None

    if not is_exclusively_used(var_node, var_eps_node):
        return False, None

    # Check variance calculation
    if (
        not check_var_op(var_node)
        or get_input_node(var_node, 0) != inputs
        or get_input_node(var_node, 1) != [-1]
        or get_input_kw_node(var_node, "keepdim") != True
    ):
        return False, None
    
    # Check unbiased parameter (either unbiased=False or correction=0)
    unbiased = get_input_kw_node(var_node, "unbiased")
    correction = get_input_kw_node(var_node, "correction")
    if not (unbiased == False or correction == 0):
        return False, None

    return True, (inputs, eps)


def _is_casted_layernorm(node: fx.Node):
    if not is_type_cast(node):
        return False
    inner = get_input_node(node, 0)
    if not is_exclusively_used(inner, node):
        return False
    
    # Direct cast of fused LayerNorm module
    if inner.op == "call_module" and isinstance(getattr(inner.graph.owning_module, inner.target), DefaultLayerNorm):
        inputs = inner.args[0]
        if not is_type_cast(inputs):
            return False
        real_inputs = get_input_node(inputs, 0)
        return real_inputs.meta["val"].dtype == node.meta["val"].dtype
    
    # Cast of bias addition (LayerNorm + bias case)
    if check_add_op(inner):
        add_input0 = get_input_node(inner, 0)
        add_input1 = get_input_node(inner, 1)
        
        # Check if one of the add inputs is a fused LayerNorm module
        layernorm_node = None
        bias = None
        
        if (add_input0.op == "call_module" and 
            isinstance(getattr(add_input0.graph.owning_module, add_input0.target), DefaultLayerNorm)):
            layernorm_node = add_input0
            bias = add_input1
        elif (add_input1.op == "call_module" and 
              isinstance(getattr(add_input1.graph.owning_module, add_input1.target), DefaultLayerNorm)):
            layernorm_node = add_input1
            bias = add_input0
            
        if layernorm_node is not None:
            inputs = layernorm_node.args[0]
            if not is_type_cast(inputs):
                return False
            real_inputs = get_input_node(inputs, 0)
            return real_inputs.meta["val"].dtype == node.meta["val"].dtype

    return False


def _is_weighted_layernorm(node: fx.Node):
    # Pattern: unaffined_layernorm * weight
    if check_mul_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_unaffined(node):
            if not isinstance(node, fx.Node) or node.op != "call_module":
                return False
            target_mod = getattr(node.graph.owning_module, node.target)
            return isinstance(target_mod, DefaultLayerNorm) and node.args[1] is None and node.args[2] is None

        if _is_unaffined(arg0):
            return True, [arg0, arg1]
        elif _is_unaffined(arg1):
            return True, [arg1, arg0]

    return False, None


def _is_layernorm(node: fx.Node):
    # Pattern: weighted_layernorm + bias
    if check_add_op(node):
        arg0 = node.args[0]
        arg1 = node.args[1]

        def _is_weighted(node):
            if not isinstance(node, fx.Node) or node.op != "call_module":
                return False
            target_mod = getattr(node.graph.owning_module, node.target)
            return isinstance(target_mod, DefaultLayerNorm) and node.args[1] is not None and node.args[2] is None

        if _is_weighted(arg0):
            return True, [arg0, arg1]
        elif _is_weighted(arg1):
            return True, [arg1, arg0]

    return False, None


class FusedLayerNorm(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_layer_norm"):
            graph_module.add_module("fused_layer_norm", DefaultLayerNorm())
        
        for node in graph_module.graph.nodes:
            # Check for full layernorm pattern (with bias): normalized * weight + bias
            if check_add_op(node):
                # This could be: mul_node + bias  
                mul_node = get_input_node(node, 0)
                bias = get_input_node(node, 1)
                
                # Handle bias position flexibility
                if isinstance(bias, fx.Node):
                    mul_node, bias = bias, mul_node
                
                if check_mul_op(mul_node):
                    # This could be: normalized * weight
                    normalized_node = get_input_node(mul_node, 0)
                    weight = get_input_node(mul_node, 1)
                    
                    # Handle weight position flexibility
                    if isinstance(weight, fx.Node):
                        normalized_node, weight = weight, normalized_node
                        
                    # Check if normalized_node is unaffined layernorm
                    matched, layernorm_param = _is_unaffined_layernorm(normalized_node)
                    if matched:
                        inputs, eps = layernorm_param
                        if eps is None:
                            eps = 1e-5
                        
                        with graph_module.graph.inserting_before(node):
                            layernorm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, bias, eps))

                        node.replace_all_uses_with(layernorm_node, propagate_meta=True)
                        is_modified = True
                        continue
            
            # Check for weighted layernorm pattern: normalized * weight
            elif check_mul_op(node):
                normalized_node = get_input_node(node, 0)
                weight = get_input_node(node, 1)
                
                # Handle weight position flexibility
                if isinstance(weight, fx.Node):
                    normalized_node, weight = weight, normalized_node
                    
                matched, layernorm_param = _is_unaffined_layernorm(normalized_node)
                if matched:
                    inputs, eps = layernorm_param
                    if eps is None:
                        eps = 1e-5
                    
                    with graph_module.graph.inserting_before(node):
                        layernorm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, None, eps))

                    node.replace_all_uses_with(layernorm_node, propagate_meta=True)
                    is_modified = True
                    continue
            
            # Check for unaffined layernorm pattern
            matched, layernorm_param = _is_unaffined_layernorm(node)
            if matched:
                inputs, eps = layernorm_param
                if eps is None:
                    eps = 1e-5

                with graph_module.graph.inserting_before(node):
                    layernorm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, None, None, eps))

                node.replace_all_uses_with(layernorm_node, propagate_meta=True)
                is_modified = True
            
            # Check for module-based patterns (for second-stage optimization)
            elif check_mul_op(node):
                res, layernorm_param = _is_weighted_layernorm(node)
                if res:
                    unaffined, weight = layernorm_param
                    inputs, _, _, eps = unaffined.args
                    with graph_module.graph.inserting_before(node):
                        layernorm_node = graph_module.graph.call_module("fused_layer_norm", (inputs, weight, None, eps))

                    node.replace_all_uses_with(layernorm_node, propagate_meta=True)
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
            if _is_casted_layernorm(node):
                inner = get_input_node(node, 0)
                
                # Case 1: Direct cast of fused LayerNorm module
                if inner.op == "call_module" and isinstance(getattr(inner.graph.owning_module, inner.target), DefaultLayerNorm):
                    layernorm_node = inner
                    inputs, weight, bias, eps = layernorm_node.args
                    real_inputs = get_input_node(inputs, 0)
                    with graph_module.graph.inserting_before(node):
                        new_layernorm = graph_module.graph.call_module("fused_layer_norm", (real_inputs, weight, bias, eps))

                    node.replace_all_uses_with(new_layernorm)
                    is_modified = True
                
                # Case 2: Cast of bias addition (LayerNorm + bias case)
                elif check_add_op(inner):
                    add_input0 = get_input_node(inner, 0)
                    add_input1 = get_input_node(inner, 1)
                    
                    # Find LayerNorm node and bias
                    layernorm_node = None
                    bias = None
                    
                    if (add_input0.op == "call_module" and 
                        isinstance(getattr(add_input0.graph.owning_module, add_input0.target), DefaultLayerNorm)):
                        layernorm_node = add_input0
                        bias = add_input1
                    elif (add_input1.op == "call_module" and 
                          isinstance(getattr(add_input1.graph.owning_module, add_input1.target), DefaultLayerNorm)):
                        layernorm_node = add_input1
                        bias = add_input0
                        
                    if layernorm_node is not None:
                        inputs, weight, _, eps = layernorm_node.args  # Note: ignore the None bias from LayerNorm
                        real_inputs = get_input_node(inputs, 0)
                        with graph_module.graph.inserting_before(node):
                            new_layernorm = graph_module.graph.call_module("fused_layer_norm", (real_inputs, weight, bias, eps))

                        node.replace_all_uses_with(new_layernorm)
                        is_modified = True
        
        return is_modified