import operator

import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup

from ..utils.check_ops import check_op, is_firstly_used
from ..utils.combo_utils import COMBINABLE_POI_OP_IDX, COMBINE_WIDTH, ComboManager
from ..utils.shape_utils import SymShapeManager

aten = torch.ops.aten


class CombinePointwiseSink(Pattern):
    _opt_level = OptLevel.level1
    _pattern_group = PatternGroup.GROUP1  # Note: This pattern should be applied after folding patterns
    _support_stages = [FxStage.inference, FxStage.pregrad]

    """
    input_1 -> poi_1 ----\
    input_2 -> poi_2 ------> stack/cat -> output
    input_3 -> poi_3 ----/  /
    other   ---------------/
    ---->
    input_1 ----\
    input_2 -----> stack -> poi -> unbind-> stack/cat -> output
    input_3 ----/                          /
    other   ------------------------------/

    N*poi + 1*cat -> 1*stack + 1+poi + 1*unbind + 1*cat
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        shape_manager = SymShapeManager(graph_module.graph)
        for node in reversed(graph_module.graph.nodes):
            if (
                check_op(node, aten.stack.default)
                or check_op(node, aten.cat.default)
                or check_op(node, aten.concat.default)
                or node.op == "output"
            ):
                if len(node.args[0]) < COMBINE_WIDTH:
                    continue
            else:
                continue

            if check_op(node, aten.cat.default) or check_op(node, aten.concat.default):
                cat_dim = 0
                if len(node.args) > 1:
                    cat_dim = node.args[1]
                elif "dim" in node.kwargs:
                    cat_dim = node.kwargs["dim"]
            else:
                cat_dim = None

            extra_shape_check = node.op == "output"

            for poi_op, combinable_argidxs in COMBINABLE_POI_OP_IDX:
                for combinable_argidx in combinable_argidxs:
                    combo_manager = ComboManager(
                        graph_module, poi_op, combinable_argidx, cat_dim, extra_shape_check, shape_manager
                    )
                    for arg in node.args[0]:
                        if isinstance(arg, fx.Node) and check_op(arg, poi_op) and is_firstly_used(arg, node):
                            combo_manager.try_add_candidate(arg)

                    with graph_module.graph.inserting_before(node):
                        for (
                            orig_args,
                            orig_results,
                            combined_arg,
                            combined_result,
                            split_node,
                        ) in combo_manager.generate_combined_results():
                            changed = True

                            for combined_idx, orig_result in enumerate(orig_results):
                                split_result = graph_module.graph.create_node(
                                    "call_function",
                                    operator.getitem,
                                    args=(split_node, combined_idx),
                                    name=orig_result.name + "_combined",
                                )
                                orig_result.replace_all_uses_with(split_result)
                                graph_module.graph.erase_node(orig_result)

        return changed
