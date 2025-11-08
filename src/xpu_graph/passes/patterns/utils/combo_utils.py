import os
from typing import Optional, Tuple, Union

import torch
from torch import fx, nn

from xpu_graph.utils import __XPU_GRAPH_ENVS__, logger

from .check_ops import check_op
from .shape_utils import SymShapeManager, same_shape


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


DEFAULT_COMBINE_POI_WIDTH = "3"
DEFAULT_COMBINABLE_POI_OP_IDX = "aten.mul.Tensor:0,1;aten.add.Tensor:0,1;aten.where.self:1,2;"

COMBINE_POI_WIDTH = max(int(os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_width, DEFAULT_COMBINE_POI_WIDTH)), 2)
COMBINABLE_POI_OP_IDX = _fetch_combinable_ops_idx(
    DEFAULT_COMBINABLE_POI_OP_IDX + os.getenv(__XPU_GRAPH_ENVS__.pointwise_combine_ops_idx, "")
)

DEFAULT_COMBINE_MM_WIDTH = "4"
COMBINE_MM_WIDTH = max(int(os.getenv(__XPU_GRAPH_ENVS__.matmul_combine_width, DEFAULT_COMBINE_MM_WIDTH)), 4)


class ComboPoiManager:
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
                if self.shape_mgr is None or self.shape_mgr.get_shape_val(combined_value) is None:
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
                if len(orig_results) < COMBINE_POI_WIDTH:
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
                    combined_sizes = None
                    combined_arg = self.graph_module.graph.call_function(
                        torch.ops.aten.stack.default,
                        args=(orig_args,),
                    )
                else:
                    combined_sizes = [result.meta["val"].shape[self.concat_dim] for result in orig_results]
                    combined_sizes = self.shape_mgr.rebind_shape(combined_sizes)
                    if any(s is None for s in combined_sizes):
                        continue
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
                    split_node = self.graph_module.graph.call_function(
                        torch.ops.aten.split_with_sizes.default,
                        args=(combined_result, combined_sizes),
                        kwargs={"dim": self.concat_dim},
                    )

                replace_groups.append((orig_args, orig_results, combined_arg, combined_result, split_node))
        return replace_groups


def partially_topo_sort(node: fx.Node, insert_after: Optional[fx.Node] = None):
    if insert_after is None or insert_after < node:
        insert_after = node
    import queue

    que = queue.Queue()
    que.put(node)
    while not que.empty():
        cur = que.get()
        for user in cur.users:
            if user < cur:
                insert_after.append(user)
                que.put(user)


def extract_nodes_from_args_kwargs(args, kwargs):
    """
    从给定的 args 和 kwargs 中递归提取所有 fx.Node 实例。
    """
    nodes = []

    def recurse(item):
        if isinstance(item, fx.Node):
            nodes.append(item)
        elif isinstance(item, (list, tuple)):
            for elem in item:
                recurse(elem)
        elif isinstance(item, dict):
            for value in item.values():
                recurse(value)
        # 其他类型（如 int、float、str 等）不处理

    recurse(args)
    recurse(kwargs)
    return nodes


def get_ancestors(node):
    """
    找给定node的所有祖先
    """
    stack = [node]
    ancestors = []
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        if node is None:
            continue
        if node.op == "placeholder":
            continue
        ancestors.append(node)
        stack += extract_nodes_from_args_kwargs(node.args, node.kwargs)
    if len(ancestors) > 0:
        # remove node
        ancestors = ancestors[1:]
    return ancestors


def find_dep(nodes, dep_func):
    """
    给定依赖函数和节点队列, 返回分组好的节点.
    """
    groups = []

    for node in nodes:
        placed = False
        for group in groups:
            if any(dep_func(node, other) for other in group):
                continue
            group.append(node)
            placed = True
            break
        if not placed:
            groups.append([node])

    return groups
