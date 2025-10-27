import torch.fx as fx

from xpu_graph import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import (
    get_input_node,
    is_exclusively_used,
    is_type_cast,
)
from xpu_graph.passes.patterns.utils.default_replacements import DefaultRMSNorm

from .check_npu_ops import check_npu_dtype_cast_op


def _is_casted_rmsnorm(node: fx.Node):
    if not check_npu_dtype_cast_op(node) and not is_type_cast(node):
        return False
    inner = get_input_node(node, 0)
    if not is_exclusively_used(inner, node):
        return False
    if inner.op == "call_module" and isinstance(getattr(inner.graph.owning_module, inner.target), DefaultRMSNorm):
        inputs = inner.args[0]
        if not check_npu_dtype_cast_op(inputs) and not is_type_cast(node):
            return False
        real_inputs = get_input_node(inputs, 0)
        return real_inputs.meta["val"].dtype == node.meta["val"].dtype

    return False


# FIXME: This pattern exists because torch-npu under inference mode dispatchs Tensor.to.dtype to npu_dtype_cast
#        It should be removed once the dispatch_fx mechanism of xpu_graph is refactored
class RemoveRMSNormNpuCast(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_rms_norm"):
            graph_module.add_module("fused_rms_norm", DefaultRMSNorm())
        for node in reversed(graph_module.graph.nodes):
            if _is_casted_rmsnorm(node):
                rms_node = get_input_node(node, 0)
                inputs, weight, eps = rms_node.args
                real_inputs = get_input_node(inputs, 0)
                with graph_module.graph.inserting_before(node):
                    new_rmsnorm = graph_module.graph.call_module("fused_rms_norm", (real_inputs, weight, eps))

                node.replace_all_uses_with(new_rmsnorm)
                is_modified = True
        return is_modified
