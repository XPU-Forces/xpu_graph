import os
from typing import Optional

import torch
import torch.fx as fx

from xpu_graph.utils import __XPU_GRAPH_ENVS__, logger

from .check_ops import check_op
from .shape_utils import SymShapeManager, same_shape, same_shape_except_dim


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


DEFAULT_COMBINE_WIDTH = "3"
DEFAULT_COMBINABLE_POI_OP_IDX = "aten.mul.Tensor:0,1;aten.add.Tensor:0,1;aten.where.self:1,2;"

COMBINE_WIDTH = max(int(os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_width, DEFAULT_COMBINE_WIDTH)), 2)
COMBINABLE_POI_OP_IDX = _fetch_combinable_ops_idx(
    DEFAULT_COMBINABLE_POI_OP_IDX + os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_ops_idx, "")
)


class ComboManager:
    def __init__(
        self,
        gm: fx.GraphModule,
        combo_op: torch._ops.OpOverload,
        combo_argidx: int,
        concat_dim: Optional[int] = None,
        extra_shape_check: bool = False,
        shape_mgr: Optional[SymShapeManager] = None,
    ):
        self.graph_module = gm
        self.combo_op = combo_op
        self.combo_argidx = combo_argidx
        self.concat_dim = concat_dim
        self.shared_to_combinelists = {}
        self.extra_shape_check = extra_shape_check
        self.shape_mgr = shape_mgr

    def check_compatible_shape(self, shape0, shape1):
        if len(shape0) != len(shape1):
            return False

        ndim = len(shape0)
        if self.concat_dim is not None:
            return -ndim <= self.concat_dim < ndim and same_shape_except_dim(shape0, shape1, self.concat_dim)
        if same_shape(shape0, shape1):
            return True
        # find extra chance to concat shapes
        for maybe_concat_dim in range(ndim):
            if same_shape_except_dim(shape0, shape1, maybe_concat_dim):
                return True
        return False

    def _is_unbacked_symshape(self, shape, dim):
        if isinstance(shape[dim], torch.SymInt):
            if self.shape_mgr is None or self.shape_mgr.get_shape_val(shape[dim]) is None:
                return True
        return False

    def _find_max_concat_list(self, candidate_nodes):
        n0 = candidate_nodes[0]
        ndim = len(n0.meta["val"].shape)
        max_concat_dim = 0
        max_concat_list = []
        for maybe_concat_dim in range(ndim):
            # symbolic shape cannot be concated as split_with_sizes needs concrete shape
            if self._is_unbacked_symshape(n0.meta["val"].shape, maybe_concat_dim):
                continue
            cur_list = [n0]
            for n in candidate_nodes[1:]:
                if same_shape_except_dim(n0.meta["val"].shape, n.meta["val"].shape, maybe_concat_dim):
                    # symbolic shape cannot be concated as split_with_sizes needs concrete shape
                    if self._is_unbacked_symshape(n.meta["val"].shape, maybe_concat_dim):
                        continue
                    cur_list.append(n)
            if len(cur_list) > len(max_concat_list):
                max_concat_dim = maybe_concat_dim
                max_concat_list = cur_list
        return max_concat_dim, max_concat_list

    def try_add_candidate(self, result_node):
        if (
            not isinstance(result_node, fx.Node)
            or not check_op(result_node, self.combo_op)
            or not isinstance(result_node.args[self.combo_argidx], fx.Node)
            or not isinstance(result_node.args[self.combo_argidx].meta["val"], torch.Tensor)
        ):
            return False

        # currently, the combination rule is strict:
        # only when the original node is not broadcasted and has same requires_grad
        combined_value = result_node.args[self.combo_argidx].meta["val"]
        if not same_shape(combined_value.shape, result_node.meta["val"].shape):
            return False

        if self.concat_dim is not None and self._is_unbacked_symshape(combined_value.shape, self.concat_dim):
            # symbolic shape cannot be concated as split_with_sizes needs concrete shape
            return False

        other_args_kwargs = (
            tuple(result_node.args[: self.combo_argidx]) + tuple(result_node.args[self.combo_argidx + 1 :]),
            result_node.kwargs,
        )

        if other_args_kwargs in self.shared_to_combinelists:
            for example_val, combine_list in self.shared_to_combinelists[other_args_kwargs]:
                combinable = (
                    combined_value.device == example_val.device
                    and combined_value.dtype == example_val.dtype
                    and combined_value.requires_grad == example_val.requires_grad
                )
                if self.extra_shape_check:
                    # Note: actually, this check is only for combining within outputs/inputs nodes
                    # because they have no shape constraints
                    combinable = combinable and self.check_compatible_shape(combined_value.shape, example_val.shape)

                if combinable:
                    combine_list.append(result_node)
                    return True
            self.shared_to_combinelists[other_args_kwargs].append((combined_value, [result_node]))
        else:
            self.shared_to_combinelists[other_args_kwargs] = [(combined_value, [result_node])]
        return True

    def generate_combined_results(self):
        replace_groups = []
        for other_args_kwargs, combinable_lists in self.shared_to_combinelists.items():
            for _, candidate_results in combinable_lists:
                if self.extra_shape_check:
                    concat_dim, orig_results = self._find_max_concat_list(candidate_results)
                else:
                    orig_results = candidate_results
                    concat_dim = self.concat_dim
                if len(orig_results) < COMBINE_WIDTH:
                    continue
                orig_args = [orig_result.args[self.combo_argidx] for orig_result in orig_results]
                do_stack = True
                if concat_dim is not None:
                    # Note: for concats, if all concated shapes are same,
                    #       we should use stack instead to avoid non-broadcastable cases
                    for orig_result in orig_results[1:]:
                        if not same_shape(
                            orig_result.meta["val"].shape[concat_dim],
                            orig_results[0].meta["val"].shape[concat_dim],
                        ):
                            do_stack = False
                            break
                if do_stack:
                    combined_sizes = None
                    combined_arg = self.graph_module.graph.call_function(
                        torch.ops.aten.stack.default,
                        args=(orig_args,),
                    )
                else:
                    combined_sizes = [result.meta["val"].shape[concat_dim] for result in orig_results]
                    combined_sizes = self.shape_mgr.rebind_shape(combined_sizes)
                    if any(s is None for s in combined_sizes):
                        continue
                    combined_arg = self.graph_module.graph.call_function(
                        torch.ops.aten.cat.default, args=(orig_args,), kwargs={"dim": concat_dim}
                    )

                shared_args, shared_kwargs = other_args_kwargs
                shared_args = (
                    tuple(shared_args[: self.combo_argidx]) + (combined_arg,) + tuple(shared_args[self.combo_argidx :])
                )
                combined_result = self.graph_module.graph.call_function(
                    self.combo_op, args=shared_args, kwargs=shared_kwargs
                )
                if do_stack:
                    split_node = self.graph_module.graph.call_function(
                        torch.ops.aten.unbind.int, args=(combined_result,)
                    )
                else:
                    split_node = self.graph_module.graph.call_function(
                        torch.ops.aten.split_with_sizes.default,
                        args=(combined_result, combined_sizes),
                        kwargs={"dim": concat_dim},
                    )

                replace_groups.append((orig_args, orig_results, combined_arg, combined_result, split_node))
        return replace_groups
