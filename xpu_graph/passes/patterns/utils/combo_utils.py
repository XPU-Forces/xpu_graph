import operator
import os

import torch
import torch.fx as fx

from xpu_graph.utils import __XPU_GRAPH_ENVS__, logger

from .check_ops import check_op
from .shape_utils import same_shape


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
    def __init__(self, gm, combo_op, combo_argidx, concat_dim=None, extra_shape_check=False):
        self.graph_module = gm
        self.combo_op = combo_op
        self.combo_argidx = combo_argidx
        self.concat_dim = concat_dim
        self.shared_to_combinelists = {}
        self.extra_shape_check = extra_shape_check

    def check_compatible_shape(self, shape0, shape1):
        if self.concat_dim is None:
            return same_shape(shape0, shape1)
        elif -len(shape0) <= self.concat_dim < len(shape0) and -len(shape1) <= self.concat_dim < len(shape1):
            shape0 = list(shape0)
            shape1 = list(shape1)
            shape0[self.concat_dim] = 1
            shape1[self.concat_dim] = 1
            return same_shape(shape0, shape1)
        else:
            return False

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

        if self.concat_dim is not None:
            if isinstance(combined_value.shape[self.concat_dim], torch.SymInt):
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
            for _, orig_results in combinable_lists:
                if len(orig_results) < COMBINE_WIDTH:
                    continue
                orig_args = [orig_result.args[self.combo_argidx] for orig_result in orig_results]
                do_stack = True
                if self.concat_dim is not None:
                    # Note: for concats, if all concated shapes are same,
                    #       we should use stack instead to avoid non-broadcastable cases
                    for orig_result in orig_results[1:]:
                        if not same_shape(
                            orig_result.meta["val"].shape[self.concat_dim],
                            orig_results[0].meta["val"].shape[self.concat_dim],
                        ):
                            do_stack = False
                            break
                if do_stack:
                    combined_arg = self.graph_module.graph.call_function(
                        torch.ops.aten.stack.default,
                        args=(orig_args,),
                    )
                else:
                    combined_arg = self.graph_module.graph.call_function(
                        torch.ops.aten.cat.default, args=(orig_args,), kwargs={"dim": self.concat_dim}
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
                    combined_sizes = [result.meta["val"].shape[self.concat_dim] for result in orig_results]
                    split_node = self.graph_module.graph.call_function(
                        torch.ops.aten.split_with_sizes.default,
                        args=(combined_result, combined_sizes),
                        kwargs={"dim": self.concat_dim},
                    )

                replace_groups.append((orig_args, orig_results, combined_arg, combined_result, split_node))
        return replace_groups
