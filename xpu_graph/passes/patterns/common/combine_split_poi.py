import os

import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import __XPU_GRAPH_ENVS__, logger

from ..utils.check_ops import check_op, is_firstly_used

aten = torch.ops.aten
import operator

DEFAULT_COMBINE_WIDTH = "3"
DEFAULT_COMBINABLE_POI_OP_IDX = "aten.mul.Tensor:0,1;aten.add.Tensor:0,1;aten.where.self:1,2;"


def _fetch_combinable_ops_idx(config_str):
    extra_combinable_ops_idx = []
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
        extra_combinable_ops_idx.append((target, combine_idxs))
    return extra_combinable_ops_idx


COMBINE_WIDTH = max(int(os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_width, DEFAULT_COMBINE_WIDTH)), 2)
COMBINABLE_POI_OP_IDX = _fetch_combinable_ops_idx(
    DEFAULT_COMBINABLE_POI_OP_IDX + os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_ops_idx, "")
)


def try_add_parallel_lists(result_node, combined_idx, shared_to_combinelists, concat_dim=None):
    if not isinstance(result_node.args[combined_idx], fx.Node) or not isinstance(
        result_node.args[combined_idx].meta["val"], torch.Tensor
    ):
        return

    # currently, the combination rule is strict:
    # only when the original node is not broadcasted and has same requires_grad
    combined_value = result_node.args[combined_idx].meta["val"]
    if combined_value.shape != result_node.meta["val"].shape:
        return

    if concat_dim is not None:
        if isinstance(combined_value.shape[concat_dim], torch.SymInt):
            # symbolic shape cannot be concated as split_with_sizes needs concrete shape
            return

    other_args_kwargs = (
        tuple(result_node.args[:combined_idx]) + tuple(result_node.args[combined_idx + 1 :]),
        result_node.kwargs,
    )

    if other_args_kwargs in shared_to_combinelists:
        for example_val, combine_list in shared_to_combinelists[other_args_kwargs]:
            if (
                combined_value.device == example_val.device
                and combined_value.dtype == example_val.dtype
                and combined_value.requires_grad == example_val.requires_grad
                # Note: no need to check shape as is resricted by broadcast check
                # and combined_value.shape == example_val.shape
            ):
                combine_list.append(result_node)
                return
        shared_to_combinelists[other_args_kwargs].append((combined_value, [result_node]))
    else:
        shared_to_combinelists[other_args_kwargs] = [(combined_value, [result_node])]


class CombineSplitPoi(Pattern):
    _opt_level = OptLevel.level1
    _pattern_group = PatternGroup.GROUP1  # Note: This pattern should be applied after folding patterns
    _support_stages = [FxStage.inference, FxStage.pregrad]

    """
                     /-----> mul -----> o1
    input ---> split ------> mul -----> o2
                     \-----> mul -----> o3
    ->
                                /-----> o1
    input ---> mul -----> split ------> o2
                                \-----> o3

    split + N*poi -> 1+poi + 1*split
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.split.Tensor
        ]
        for node in candidates:
            for poi_op, combinable_argidxs in COMBINABLE_POI_OP_IDX:
                for combinable_argidx in combinable_argidxs:
                    shared_to_combinelists = {}
                    for geti in node.users:
                        if geti.target != operator.getitem:
                            continue
                        if len(geti) != 1:
                            continue
                        for poi_node in geti.users:
                            if isinstance(poi_node, fx.Node) and check_op(
                                poi_node, poi_op
                            ):  # and is_firstly_used(arg, node):
                                try_add_parallel_lists(
                                    poi_node, combinable_argidx, shared_to_combinelists
                                )  # , cat_dim)

                    for shared_args_kwargs, combinable_lists in shared_to_combinelists.items():
                        for _, combinable_results in combinable_lists:
                            if len(combinable_results) == len(node.users):
                                changed = True
                                shared_args, shared_kwargs = shared_args_kwargs
                                shared_args = (
                                    tuple(shared_args[:combinable_argidx])
                                    + (node.args[0],)
                                    + tuple(shared_args[combinable_argidx:])
                                )
                                with graph_module.graph.inserting_after(node.args[0]):
                                    combined_poi = graph_module.graph.call_function(
                                        poi_op, args=shared_args, kwargs=shared_kwargs
                                    )
                                    input_args = list(node.args)
                                    input_args[0] = combined_poi
                                    node.args = tuple(input_args)
                                for n in combinable_results:
                                    n.replace_all_uses_with(n.args[0])
        return changed
