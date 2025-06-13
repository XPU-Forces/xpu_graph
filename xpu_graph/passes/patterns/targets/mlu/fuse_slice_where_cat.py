from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import (
    check_cat_op,
    check_slice_op,
    check_stack_op,
    check_where_op,
    check_zeros_op,
)
from ...utils.submodule_manager import register_new_submodule
from .triton_kernel.fused_slice_where_cat import fuse_slice_where_cat


class FusedSliceWhereCatReplacement(nn.Module):
    def __init__(self, slice_params):
        super().__init__()
        device = torch.mlu.current_device()
        self.slice_param_tensor = torch.tensor(slice_params, dtype=torch.int32, device="mlu:" + str(device))
        self.slice_num = int(len(slice_params))

    def forward(
        self,
        where_input,
        slice_input,
        zeros_param,
        unmatched_nodes,
        cat_dim,
        slice_dim,
    ):
        cat_matched = fuse_slice_where_cat(
            where_input,
            slice_input,
            self.slice_param_tensor,
            zeros_param[1],
            self.slice_num,
        )
        if unmatched_nodes:
            unmatched_nodes = unmatched_nodes + [cat_matched]
            output = torch.cat(unmatched_nodes, dim=cat_dim)
            return output
        else:
            return cat_matched


def find_matching_nodes(cat_node):
    cat_inputs = cat_node.args[0]
    last_input = cat_inputs[-1]

    if not check_where_op(last_input):
        return False, ()
    matched_nodes = [last_input]

    if not hasattr(last_input, "args") or len(last_input.args) != 3:
        return False, ()

    condition, x, y = last_input.args[:3]
    if not check_zeros_op(x):
        return False, ()
    if not check_slice_op(y):
        return False, ()

    zeros_param = x.args[0]
    slice_input = y.args[0]
    slice_dim = y.args[1]
    slice_params = [y.args[2]]

    for node in reversed(cat_inputs[:-1]):
        if not hasattr(node, "args") or len(node.args) != 3:
            break
        if not check_where_op(node):
            break
        condition_, x_, y_ = node.args[:3]
        if not check_slice_op(y_):
            break

        if condition_ == condition and x_ == x and y_.args[0] == y.args[0] and y_.args[1] == y.args[1] == 1:
            matched_nodes.append(node)
            slice_params.append(y_.args[2])
        else:
            break

    if len(matched_nodes) < 2:
        return False, ()

    unmatched_nodes = [node for node in cat_inputs if node not in matched_nodes[::-1]]
    return True, (
        condition,
        slice_input,
        zeros_param,
        unmatched_nodes,
        slice_dim,
        slice_params[::-1],
    )


def _is_slice_where_cat(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    is_cat, cat_dim = check_cat_op(node)
    if not is_cat:
        return False, (), ()
    is_match, param_tuple = find_matching_nodes(node)
    if not is_match:
        return False, (), ()

    where_input_shape = param_tuple[0].meta["val"].shape
    slice_input_shape = param_tuple[1].meta["val"].shape
    if len(where_input_shape) != 2 or len(slice_input_shape) != 2:
        return False, (), ()
    if where_input_shape[1] != 1:
        return False, (), ()
    if cat_dim != 1 and cat_dim != -1:
        return False, (), ()
    if param_tuple[4] != 1 and param_tuple[4] != -1:
        return False, (), ()

    return True, param_tuple, cat_dim


def _is_stack_to_cat(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node], Optional[fx.Node], Optional[fx.Node]]:
    is_cat, cat_dim = check_cat_op(node)
    if not is_cat:
        return False, ()
    cat_inputs = node.args[0]
    last_input = cat_inputs[-1]
    stack_inputs = []
    if len(last_input.users) == 1:
        return False, ()
    for key, _ in last_input.users.items():
        if check_stack_op(key) and len(key.args) == 1 and key.args[0] == cat_inputs:
            shape = last_input.meta["val"].shape
            input_nums = len(cat_inputs)
            return True, (key, cat_inputs, shape, input_nums)
    return False, ()


class FusedSliceWhereCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        return False
        for node in reversed(graph_module.graph.nodes):
            is_change_stack, stack_params = _is_stack_to_cat(node)
            is_match, param_tuple, cat_dim = _is_slice_where_cat(node)
            if is_match:
                (
                    where_input,
                    slice_input,
                    zeros_param,
                    unmatched_nodes,
                    slice_dim,
                    slice_params,
                ) = param_tuple

                if is_change_stack:
                    with graph_module.graph.inserting_before(stack_params[0]):
                        cat_node = graph_module.graph.call_function(
                            torch.ops.aten.cat.default,
                            args=(stack_params[1], -1),
                            kwargs={},
                        )
                        reshape_node = graph_module.graph.call_function(
                            torch.ops.aten.view.default,
                            args=(
                                cat_node,
                                [
                                    stack_params[2][0],
                                    stack_params[3],
                                    stack_params[2][1],
                                ],
                            ),
                            kwargs={},
                        )
                        permute_node = graph_module.graph.call_function(
                            torch.ops.aten.permute.default,
                            args=(reshape_node, [1, 0, 2]),
                            kwargs={},
                        )
                        stack_params[0].replace_all_uses_with(permute_node)
                        graph_module.graph.erase_node(stack_params[0])

                with graph_module.graph.inserting_before(node):
                    module_name = register_new_submodule(
                        graph_module,
                        "mlu_triton_slice_where_cat",
                        FusedSliceWhereCatReplacement,
                        args=(slice_params,),
                    )
                    new_node = graph_module.graph.call_module(
                        module_name,
                        args=(
                            where_input,
                            slice_input,
                            zeros_param,
                            unmatched_nodes,
                            cat_dim,
                            slice_dim,
                        ),
                    )

                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
