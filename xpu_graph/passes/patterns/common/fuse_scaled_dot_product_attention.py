import torch
import torch.fx as fx

from xpu_graph.config import OptLevel
from xpu_graph.constant_manager import is_constant
from xpu_graph.fx_utils import FxStage, _get_wrapped_constant
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import (
    check_baddbmm_op,
    check_bmm_op,
    check_div_or_mul_op,
    check_softmax_op,
    check_sub_or_add_op,
    get_actual_node,
)
from xpu_graph.passes.patterns.utils.default_replacements import SDPAWrappedScale
from xpu_graph.passes.patterns.utils.shape_utils import (
    get_sym_shape_bindings,
    rebind_shape,
)
from xpu_graph.utils import logger


def validate_attention_shape(q_shape, k_shape, v_shape, mask_shape):
    # Since q_shape, k_shape, v_shape comes from bmm, they should be 3D and has same batch size
    # Currently, no GQA is considerred
    if not (
        len(q_shape) == 3
        and len(k_shape) == 3
        and len(v_shape) == 3
        and q_shape[0] == k_shape[0]
        and q_shape[0] == v_shape[0]
        and q_shape[-1] == k_shape[-1]
        and k_shape[1] == v_shape[1]
    ):
        return False, []
    total_bn = q_shape[0]
    try:
        q_len = q_shape[1]
        k_len = k_shape[1]
        attn_shape = torch.broadcast_shapes(mask_shape, [q_len, k_len])
    except:
        return False, []
    if len(attn_shape) > 4:
        return False, []
    elif len(attn_shape) == 4:
        batch_size, num_heads = attn_shape[:2]
    elif len(attn_shape) == 3:
        batch_size = 1
        num_heads = attn_shape[0]
    else:
        batch_size = 1
        num_heads = 1
    if num_heads == 1:
        if total_bn % batch_size != 0:
            return False, []
        else:
            return True, [batch_size, total_bn // batch_size, q_len, k_len, q_shape[-1], v_shape[-1]]
    else:
        if total_bn % num_heads != 0:
            return False, []
        else:
            return True, [total_bn // num_heads, num_heads, q_len, k_len, q_shape[-1], v_shape[-1]]


def _is_fa(node: fx.Node, is_inference: bool):
    _is_bmm, score_node, v_node = check_bmm_op(node)
    if not _is_bmm:
        return False, []
    softmax_node = get_actual_node(score_node)
    if not check_softmax_op(softmax_node):
        return False, []

    bmm_1_node = get_actual_node(softmax_node, 0)

    # (optional) find add
    mask, is_add = (None, False)
    is_bias_op, _, params = check_sub_or_add_op(bmm_1_node)
    if is_bias_op:
        mask, is_add = params
        bmm_1_node = get_actual_node(bmm_1_node, 0)

    # (optional) find div or mul
    scale, is_div = (1.0, False)
    is_scale_op, div_input_node, params = check_div_or_mul_op(bmm_1_node)
    if is_scale_op:
        scale, is_div = params
        if isinstance(scale, fx.Node) and isinstance(scale.meta["val"], torch.Tensor):
            if not scale.meta["val"].numel() == 1:
                return False, []
            if is_constant(scale):
                scale = _get_wrapped_constant(scale)

        bmm_1_node = div_input_node

    _is_bmm, q_node, k_node = check_bmm_op(bmm_1_node)
    if not _is_bmm:
        _is_baddbmm, mask, q_node, k_node = check_baddbmm_op(bmm_1_node)
        if is_bias_op or is_scale_op or not _is_baddbmm:
            logger.debug("Flash attention pass: Too many add operations")
            return False, []
        is_bias_op = True
        is_add = True

    q_shape = q_node.meta["val"].shape
    k_shape = k_node.meta["val"].shape
    k_shape = k_shape[:-2] + (k_shape[-1], k_shape[-2])

    v_shape = v_node.meta["val"].shape

    if is_bias_op and isinstance(mask, fx.Node):
        mask_shape = mask.meta["val"].shape
    else:
        mask_shape = []

    is_valid, attn_shape = validate_attention_shape(q_shape, k_shape, v_shape, mask_shape)
    if not is_valid:
        return False, []

    return True, [q_node, k_node, v_node, scale, is_div, mask, is_add, attn_shape]


class FusedSDPA(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule):
        modified = False

        # Note: this fuction should rerun every time needed, since graph may be changed elsewhere
        sym_shape_to_node_map = get_sym_shape_bindings(graph_module.graph)

        for node in reversed(graph_module.graph.nodes):
            is_inference = self._current_stage == FxStage.inference
            matched, fa_param = _is_fa(node, is_inference)
            if not matched:
                continue

            q, k, v, scale, is_div, attn_mask, is_add, attn_shape = fa_param

            # Note: if the shape is non-atomic sympy expression, it must have been computed beforehand, thus it is okay to
            # use get_shape_val to get the actual value
            rebind_attn_shape = rebind_shape(attn_shape, sym_shape_to_node_map)

            if rebind_attn_shape is None:
                continue
            batch_size, num_heads, q_len, kv_len, qk_dim, v_dim = rebind_attn_shape

            with graph_module.graph.inserting_before(node):
                is_scale_wrapped = False
                if isinstance(scale, fx.Node) and isinstance(scale.meta["val"], torch.Tensor):
                    if self._opt_level > OptLevel.level2:
                        q = graph_module.graph.call_function(
                            torch.ops.aten.div.Tensor if is_div else torch.ops.aten.mul.Tensor,
                            args=(q, scale),
                        )
                        scale = 1.0
                    elif not is_inference:
                        logger.debug("Note: we currently do not provide SDPAWrappedScale.backward")
                        continue
                    else:
                        logger.warning("Unwrap scale for scaled_dot_product_attention, which may introduce extra sync")
                        is_scale_wrapped = True

                if is_div:
                    if is_scale_wrapped:
                        scale = graph_module.graph.call_function(
                            torch.ops.aten.reciprocal.default,
                            args=(scale,),
                        )
                    elif isinstance(scale, fx.Node):
                        import operator

                        scale = graph_module.graph.call_function(
                            operator.truediv,
                            args=(1.0, scale),
                        )
                    else:
                        scale = 1.0 / float(scale)
                # If k has been transposed, the extra transpose will be folded afterwards
                k = graph_module.graph.call_function(
                    torch.ops.aten.transpose.int,
                    args=(k, -1, -2),
                )
                q = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(q, (batch_size, num_heads, q_len, qk_dim)),
                )
                k = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(k, (batch_size, num_heads, kv_len, qk_dim)),
                )
                v = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(v, (batch_size, num_heads, kv_len, v_dim)),
                )
                q = graph_module.graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(q,),
                    kwargs={"memory_format": torch.contiguous_format},
                )
                k = graph_module.graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(k,),
                    kwargs={"memory_format": torch.contiguous_format},
                )
                v = graph_module.graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(v,),
                    kwargs={"memory_format": torch.contiguous_format},
                )

                if attn_mask is not None and not is_add:
                    if isinstance(attn_mask, fx.Node):
                        attn_mask = graph_module.graph.call_function(
                            torch.ops.aten.neg.default,
                            (attn_mask,),
                        )
                    else:
                        attn_mask = -attn_mask

                if is_scale_wrapped:
                    if not hasattr(graph_module, "fused_sdpa"):
                        graph_module.add_submodule("fused_sdpa", SDPAWrappedScale())
                    fused = graph_module.graph.call_module(
                        "fused_sdpa",
                        args=(q, k, v, attn_mask, scale),
                    )
                else:
                    fused = graph_module.graph.call_function(
                        torch.ops.aten.scaled_dot_product_attention.default,
                        args=(q, k, v),
                        kwargs={"attn_mask": attn_mask, "scale": scale},
                    )
                view = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    args=(fused, (num_heads, -1, v_dim)),
                )
            node.replace_all_uses_with(view)
            modified = True

        return modified
