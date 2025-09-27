import operator

import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup

from ..utils.check_ops import check_op, get_input_node
from ..utils.combo_utils import COMBINABLE_POI_OP_IDX, COMBINE_WIDTH, ComboManager

aten = torch.ops.aten


class CombinePointwiseSource(Pattern):
    _opt_level = OptLevel.level1
    _pattern_group = PatternGroup.GROUP1  # Note: This pattern should be applied after folding patterns
    _support_stages = [FxStage.inference, FxStage.pregrad]

    """
                     /-----> mul -----> o1
    input ---> split ------> mul -----> o2
                     \-----> mul -----> o3
    ->
                                                   /-----> o1
    input ---> split ---> cat --->mul -----> split ------> o2
                                                   \-----> o3

    1*split + N*poi -> 1*split + 1*stack + 1*poi + 1*split
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        for node in reversed(graph_module.graph.nodes):
            if (
                check_op(node, aten.unbind.int)
                or check_op(node, aten.split_with_sizes.default)
                or check_op(node, aten.split.Tensor)
                or check_op(node, aten.split.sizes)
            ):
                candidates = [user for user in node.users if user.target == operator.getitem]
                candidates.sort(key=lambda x: x.args[1])
                # Note: Since it is after CSE, the getitem nodes should be unique
                if len(candidates) < COMBINE_WIDTH:
                    continue
            else:
                continue

            if (
                check_op(node, aten.split_with_sizes.default)
                or check_op(node, aten.split.Tensor)
                or check_op(node, aten.split.sizes)
            ):
                cat_dim = 0
                if len(node.args) > 2:
                    cat_dim = node.args[2]
                elif "dim" in node.kwargs:
                    cat_dim = node.kwargs["dim"]
            else:
                cat_dim = None

            changed = self.process_candidates(graph_module, node, candidates, cat_dim) or changed

        placeholders = [
            node
            for node in graph_module.graph.nodes
            if node.op == "placeholder" and "val" in node.meta and isinstance(node.meta["val"], torch.Tensor)
        ]
        if len(placeholders) >= COMBINE_WIDTH:
            changed = self.process_candidates(graph_module, None, placeholders, None) or changed

        return changed

    def process_candidates(self, graph_module, source_node, candidates, cat_dim):
        changed = False
        for poi_op, combinable_argidxs in COMBINABLE_POI_OP_IDX:
            for combinable_argidx in combinable_argidxs:
                combo_manager = ComboManager(graph_module, poi_op, combinable_argidx, cat_dim)
                for cand in candidates:
                    for result_node in cand.users:
                        if get_input_node(result_node, combinable_argidx) is cand:
                            combo_manager.try_add_candidate(result_node)

                # DO NOT specify insertion point, because new nodes will be reordered soon
                with graph_module.graph.inserting_after(source_node):
                    for (
                        orig_args,
                        orig_results,
                        combined_arg,
                        combined_result,
                        split_node,
                    ) in combo_manager.generate_combined_results():
                        changed = True
                        # Note: reorder inserted nodes to keep topo
                        first_result = min(orig_results)
                        if source_node is not None:
                            for orig_arg in orig_args:
                                first_result.prepend(orig_arg)
                        first_result.prepend(combined_arg)
                        first_result.prepend(combined_result)
                        first_result.prepend(split_node)
                        first_result = None
                        with graph_module.graph.inserting_after(split_node):
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
