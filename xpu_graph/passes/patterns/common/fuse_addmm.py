import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_add_op, check_mm_op


def _is_addmm(
    node: fx.Node,
):
    if not check_add_op(node):
        return False, ()
    mm_node = node.args[0]
    bias_node = node.args[1]
    is_mm, input_node, weight_node = check_mm_op(mm_node)
    if not is_mm:
        mm_node = node.args[1]
        bias_node = node.args[0]
        is_mm, input_node, weight_node = check_mm_op(mm_node)
        if not is_mm:
            return False, ()
    if len(mm_node.users) != 1:
        return False, ()
    if node.meta["val"].shape != mm_node.meta["val"].shape:
        return False, ()

    return True, (bias_node, input_node, weight_node)


class FusedAddMM(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        for node in reversed(graph_module.graph.nodes):
            is_match, mm_inputs = _is_addmm(node)
            if is_match:
                bias_node, input_node, weight_node = mm_inputs
                with graph_module.graph.inserting_before(node):
                    if isinstance(bias_node, float) or isinstance(bias_node, int):
                        bias_node = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.scalar_tensor.default,
                            args=(bias_node,),
                            kwargs={"dtype": weight_node.meta["val"].dtype, "device": weight_node.meta["val"].device},
                            name="bias_constant",
                        )
                    addmm_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.addmm.default,
                        args=(bias_node, input_node, weight_node),
                        name="addmm_replacement",
                    )
                node.replace_all_uses_with(addmm_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
