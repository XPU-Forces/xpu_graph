import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

from xpu_graph.config import OptLevel
from xpu_graph.constant_manager import is_constant
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import (
    check_div_or_mul_op,
    check_softmax_op,
    check_sub_or_add_op,
    get_actual_node,
)
from xpu_graph.passes.patterns.utils.default_replacements import (
    ScaledDotProductAttention,
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


def _get_wrapped_constant(node: fx.Node):
    disable_fake_mode = None
    from packaging import version

    torch_version = version.parse(torch.__version__[:5])
    if torch_version < version.parse("2.5"):
        from torch.fx.experimental.proxy_tensor import (
            maybe_disable_fake_tensor_mode as disable_fake_mode,
        )
    else:
        from torch._subclasses.fake_tensor import (
            unset_fake_temporarily as disable_fake_mode,
        )
    with disable_fake_mode():
        node = getattr(node.graph.owning_module, node.target)
        return node.item()


def _is_fa(node: fx.Node):
    if node.target != "fused_bmm_replacement":
        return False, []
    trans_v = node.args[2]
    softmax_node = get_actual_node(node, 0)
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
        if isinstance(scale, fx.Node):
            if scale.meta["val"].numel() == 1 and is_constant(scale):
                scale = _get_wrapped_constant(scale)
            else:
                return False, []

        bmm_1_node = div_input_node

    if bmm_1_node.target != "fused_bmm_replacement":
        if is_bias_op or is_scale_op or bmm_1_node.target != "fused_bmm_add_replacement":
            logger.debug("Flash attention pass: Too many add operations")
            return False, []
        bias = bmm_1_node.args[3] or bmm_1_node.args[4]
        is_bias_op = bias is not None
        if is_bias_op:
            mask, is_add = (bias, True)

    trans_k = bmm_1_node.args[2]

    q_node = bmm_1_node.args[0]
    k_node = bmm_1_node.args[1]
    v_node = node.args[1]

    q_shape = q_node.meta["val"].shape
    k_shape = k_node.meta["val"].shape
    if not trans_k:
        k_shape = k_shape[:-2] + (k_shape[-1], k_shape[-2])
    v_shape = v_node.meta["val"].shape
    if trans_v:
        v_shape = v_shape[:-2] + (v_shape[-1], v_shape[-2])
    if is_bias_op and isinstance(mask, fx.Node):
        mask_shape = mask.meta["val"].shape
    else:
        mask_shape = []

    is_valid, attn_shape = validate_attention_shape(q_shape, k_shape, v_shape, mask_shape)
    if not is_valid:
        return False, []

    return True, [q_node, k_node, trans_k, v_node, trans_v, scale, is_div, mask, is_add, attn_shape]


class FusedSDPA(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule):
        if not hasattr(graph_module, "fused_sdpa"):
            graph_module.add_submodule("fused_sdpa", ScaledDotProductAttention())
        modified = False
        for node in reversed(graph_module.graph.nodes):
            matched, fa_param = _is_fa(node)
            if not matched:
                continue

            q, k, trans_k, v, trans_v, scale, is_div, attn_mask, is_add, attn_shape = fa_param
            logger.debug(f"Flash attention pass: Attention shape: {attn_shape}")
            batch_size, num_heads, q_len, kv_len, qk_dim, v_dim = attn_shape
            with graph_module.graph.inserting_before(node):
                if not trans_k:
                    k = graph_module.graph.call_function(
                        torch.ops.aten.transpose.int,
                        (k, -1, -2),
                    )
                if trans_v:
                    v = graph_module.graph.call_function(
                        torch.ops.aten.transpose.int,
                        (v, -1, -2),
                    )
                q = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    (q, (batch_size, num_heads, q_len, qk_dim)),
                )
                k = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    (k, (batch_size, num_heads, kv_len, qk_dim)),
                )
                v = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    (v, (batch_size, num_heads, kv_len, v_dim)),
                )

                if attn_mask is not None and not is_add:
                    if isinstance(attn_mask, fx.Node):
                        attn_mask = graph_module.graph.call_function(
                            torch.ops.aten.neg.default,
                            (attn_mask,),
                        )
                    else:
                        attn_mask = -attn_mask
                if is_div:
                    if isinstance(scale, fx.Node):
                        scale = graph_module.graph.call_function(
                            torch.ops.aten.reciprocal.default,
                            (scale,),
                        )
                    else:
                        scale = 1.0 / float(scale)
                if isinstance(scale, fx.Node):
                    scale = graph_module.graph.call_function(
                        torch.ops.aten.view.default,
                        (scale, []),
                    )
                fused = graph_module.graph.call_module(
                    "fused_sdpa",
                    args=(q, k, v, attn_mask, scale),
                )
                view = graph_module.graph.call_function(
                    torch.ops.aten.view.default,
                    (fused, node.meta["val"].shape),
                )
            node.replace_all_uses_with(view)
            modified = True

        return modified
