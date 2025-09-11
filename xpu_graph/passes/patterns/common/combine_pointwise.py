import os

import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import __XPU_GRAPH_ENVS__, logger

from ..utils.check_ops import check_op, is_firstly_used

aten = torch.ops.aten
import operator

DEFAULT_COMBINE_WIDTH = "3"
DEFAULT_STACKABLE_POI_OP_IDX = "aten.mul.Tensor:0,1;aten.add.Tensor:0,1;aten.where.self:1,2;"


def _fetch_extra_stackable_ops_idx(config_str):
    extra_stackable_ops_idx = []
    for combine_op_idx in config_str.split(";"):
        if combine_op_idx == "":
            continue
        combine_op, combine_idxs = combine_op_idx.split(":")
        try:
            attrs = combine_op.split(".")
            target = torch.ops
            for attr in attrs:
                target = getattr(target, attr)
        except:
            logger.warning(f"Unsupported call_function: {combine_op}")
            continue
        combine_idxs = [int(idx) for idx in combine_idxs.split(",")]
        extra_stackable_ops_idx.append((target, combine_idxs))
    return extra_stackable_ops_idx


COMBINE_WIDTH = max(int(os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_width, DEFAULT_COMBINE_WIDTH)), 2)
STACKABLE_POI_OP_IDX = _fetch_extra_stackable_ops_idx(
    DEFAULT_STACKABLE_POI_OP_IDX + os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_ops_idx, "")
)


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
            val.device == example_val.device
            and val.dtype == example_val.dtype
            and val.requires_grad == example_val.requires_grad
            and val.shape == example_val.shape
        )

    if other_args_kwargs in shared_to_stacklists:
        for example_val, stack_list in shared_to_stacklists[other_args_kwargs]:
            if is_stackable(stacked_value, example_val):
                stack_list.append(result_node)
                return
        shared_to_stacklists[other_args_kwargs].append((stacked_value, [result_node]))
    else:
        shared_to_stacklists[other_args_kwargs] = [(stacked_value, [result_node])]


class CombinePointwiseSameShape(Pattern):
    _opt_level = OptLevel.level1
    """
    input_1 -> poi_1 ----\
    input_2 -> poi_2 ------> cat/stack/output -> output
    input_3 -> poi_3 ----/  /
    other   ---------------/
    ---->
    input_1 ----\
    input_2 -----> stack -> poi -> unbind-> cat/stack/output -> output
    input_3 ----/                          /
    other   ------------------------------/

    The mask should be same
    The other arg should be const
    N*poi + 1*cat -> 1*stack + 1+poi + 1*cat
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        for node in reversed(graph_module.graph.nodes):
            if node.op == "output" or check_op(node, aten.cat.default) or check_op(node, aten.stack.default):
                parallel_inputs_argidx = 0
                if len(node.args[parallel_inputs_argidx]) < COMBINE_WIDTH:
                    continue
            else:
                continue

            for poi_op, stackable_argidxs in STACKABLE_POI_OP_IDX:
                for stacked_argidx in stackable_argidxs:
                    shared_to_stacklists = {}
                    for arg in node.args[parallel_inputs_argidx]:
                        if isinstance(arg, fx.Node) and check_op(arg, poi_op) and is_firstly_used(arg, node):
                            try_add_stackable_lists(arg, stacked_argidx, shared_to_stacklists)

                    for shared_args_kwargs, stackable_lists in shared_to_stacklists.items():
                        for _, stackable_results in stackable_lists:
                            if len(stackable_results) >= COMBINE_WIDTH:
                                changed = True
                                stackable_args = [
                                    stackable_result.args[stacked_argidx] for stackable_result in stackable_results
                                ]
                                with graph_module.graph.inserting_before(node):
                                    stacked_arg = graph_module.graph.call_function(
                                        aten.stack.default, args=(stackable_args,), kwargs={"dim": 0}
                                    )
                                    shared_args, shared_kwargs = shared_args_kwargs
                                    shared_args = (
                                        tuple(shared_args[:stacked_argidx])
                                        + (stacked_arg,)
                                        + tuple(shared_args[stacked_argidx:])
                                    )
                                    stacked_poi = graph_module.graph.call_function(
                                        poi_op, args=shared_args, kwargs=shared_kwargs
                                    )
                                    split_results = graph_module.graph.call_function(
                                        aten.unbind.int, args=(stacked_poi,), kwargs={"dim": 0}
                                    )
                                    for stacked_idx, stackable_result in enumerate(stackable_results):
                                        split_result = graph_module.graph.call_function(
                                            operator.getitem, args=(split_results, stacked_idx)
                                        )
                                        stackable_result.replace_all_uses_with(split_result)
                                        graph_module.graph.erase_node(stackable_result)

        return changed
