import os

import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import __XPU_GRAPH_ENVS__

from ..utils.check_ops import check_op, is_exclusively_used

aten = torch.ops.aten
import operator

COMBINE_WIDTH = max(int(os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_width, "3")), 2)

STACKABLE_POI_OP_IDX = [
    (aten.mul.Tensor, 0),
    (aten.mul.Tensor, 1),
    (aten.add.Tensor, 0),
    (aten.add.Tensor, 1),
]


def try_add_stackable_lists(result_node, stacked_idx, shared_to_stacklists):
    if not isinstance(result_node.args[stacked_idx], fx.Node) or not isinstance(
        result_node.args[stacked_idx].meta["val"], torch.Tensor
    ):
        return

    # currently, the combination rule is strict:
    # only when the original node is not broadcasted and has same requires_grad
    stacked_value = result_node.args[stacked_idx].meta["val"]
    if stacked_value.shape != result_node.meta["val"].shape:
        return

    other_args_kwargs = (
        tuple(result_node.args[:stacked_idx]) + tuple(result_node.args[stacked_idx + 1 :]),
        result_node.kwargs,
    )

    def is_stackable(val, example_val):
        return (
            val.shape == example_val.shape
            and val.dtype == example_val.dtype
            and val.requires_grad == example_val.requires_grad
        )

    if other_args_kwargs in shared_to_stacklists:
        for example_val, stack_list in shared_to_stacklists[other_args_kwargs]:
            if is_stackable(stacked_value, example_val):
                stack_list.append(result_node)
                return
        shared_to_stacklists[other_args_kwargs].append((stacked_value, [result_node]))
    else:
        shared_to_stacklists[other_args_kwargs] = [(stacked_value, [result_node])]


def find_max_stackable_list(shared_to_stacklists):
    max_stackable_list = []
    max_stackable_list_len = 0
    shared_args_kwargs = None
    for maybe_shared_args_kwargs, stackable_lists in shared_to_stacklists.items():
        for _, stack_list in stackable_lists:
            if len(stack_list) > max_stackable_list_len:
                max_stackable_list = stack_list
                max_stackable_list_len = len(stack_list)
                shared_args_kwargs = maybe_shared_args_kwargs
    return max_stackable_list, shared_args_kwargs


class CombinePointwiseSameShape(Pattern):
    _opt_level = OptLevel.level1
    """
    input_1 -> poi_1 ----\
    input_2 -> poi_2 ------> cat/stack/output -> output
    input_3 -> poi_3 ----/
    other   ---------------/
    ---->
    input_1 ----\
    input_2 -----> cat -> poi -> cat -> output
    input_3 ----/              /
    other   ------------------/

    The mask should be same
    The other arg should be const
    N*poi + 1*cat -> 1*cat + 1+poi
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        for node in graph_module.graph.nodes:
            if node.op == "output" or check_op(node, aten.cat.default) or check_op(node, aten.stack.default):
                if len(node.args[0]) < COMBINE_WIDTH:
                    continue
            else:
                continue

            for poi_op, stacked_argidx in STACKABLE_POI_OP_IDX:
                shared_to_stacklists = {}
                for arg in node.args[0]:
                    if isinstance(arg, fx.Node) and is_exclusively_used(arg, node) and check_op(arg, poi_op):
                        try_add_stackable_lists(arg, stacked_argidx, shared_to_stacklists)

                stackable_results, shared_args_kwargs = find_max_stackable_list(shared_to_stacklists)
                if len(stackable_results) >= COMBINE_WIDTH and shared_args_kwargs is not None:
                    changed = True
                    stackable_args = [stackable_result.args[stacked_argidx] for stackable_result in stackable_results]
                    with graph_module.graph.inserting_before(node):
                        stacked_arg = graph_module.graph.call_function(
                            aten.stack.default, args=(stackable_args,), kwargs={"dim": 0}
                        )
                        shared_args, shared_kwargs = shared_args_kwargs
                        shared_args = (
                            tuple(shared_args[:stacked_argidx]) + (stacked_arg,) + tuple(shared_args[stacked_argidx:])
                        )
                        stacked_poi = graph_module.graph.call_function(poi_op, args=shared_args, kwargs=shared_kwargs)
                        split_results = graph_module.graph.call_function(
                            aten.unbind.int, args=(stacked_poi,), kwargs={"dim": 0}
                        )
                        for idx, stackable_result in enumerate(stackable_results):
                            split_result = graph_module.graph.call_function(operator.getitem, args=(split_results, idx))
                            stackable_result.replace_all_uses_with(split_result)
                            graph_module.graph.erase_node(stackable_result)
        return changed
