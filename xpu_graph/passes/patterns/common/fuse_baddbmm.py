import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_add_op, check_bmm_op


def _is_baddbmm(
    node: fx.Node,
):
    if not check_add_op(node):
        return False, ()
    bmm_node = node.args[0]
    bias_node = node.args[1]
    is_bmm, input_node, weight_node = check_bmm_op(bmm_node)
    if not is_bmm:
        bmm_node = node.args[1]
        bias_node = node.args[0]
        is_bmm, input_node, weight_node = check_bmm_op(bmm_node)
        if not is_bmm:
            return False, ()
    if len(bmm_node.users) != 1:
        return False, ()
    if node.meta["val"].shape != bmm_node.meta["val"].shape:
        return False, ()

    return True, (bias_node, input_node, weight_node)


class FusedBAddBMM(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        for node in reversed(graph_module.graph.nodes):
            is_match, bmm_inputs = _is_baddbmm(node)
            if is_match:
                bias_node, input_node, weight_node = bmm_inputs
                with graph_module.graph.inserting_before(node):
                    if isinstance(bias_node, (float, int)) or (
                        isinstance(bias_node, fx.Node) and not isinstance(bias_node.meta["val"], torch.Tensor)
                    ):
                        bias_node = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.scalar_tensor.default,
                            args=(bias_node,),
                            kwargs={"dtype": input_node.meta["val"].dtype, "device": input_node.meta["val"].device},
                            name="bias_constant",
                        )
                    elif bias_node.meta["val"].dtype != input_node.meta["val"].dtype:
                        bias_node = graph_module.graph.create_node(
                            op="call_function",
                            target=(
                                torch.ops.aten.to.dtype
                                if self._current_stage == FxStage.pregrad
                                else torch.ops.aten._to_copy.default
                            ),
                            args=(bias_node,),
                            kwargs={"dtype": input_node.meta["val"].dtype},
                            name="bias_cast",
                        )
                    baddbmm_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.baddbmm.default,
                        args=(bias_node, input_node, weight_node),
                        name="baddbmm_replacement",
                    )
                    if input_node.meta["val"].dtype != node.meta["val"].dtype:
                        baddbmm_node = graph_module.graph.create_node(
                            op="call_function",
                            target=(
                                torch.ops.aten.to.dtype
                                if self._current_stage == FxStage.pregrad
                                else torch.ops.aten._to_copy.default
                            ),
                            args=(baddbmm_node,),
                            kwargs={"dtype": node.meta["val"].dtype},
                            name="input_cast",
                        )
                node.replace_all_uses_with(baddbmm_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
