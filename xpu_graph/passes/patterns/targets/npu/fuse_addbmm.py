from typing import Optional

import torch
from torch import nn, fx
import torch_npu
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.check_ops import (
    check_add_op,
    check_bmm_op,
    check_view,
    check_copy,
    check_clone,
    check_op,
    get_actual_node,
    get_shape,
    get_dtype,
)

from .triton_kernel.fused_bmm import (
    triton_bmm,
)

class FusedBAddBMMReplacement(nn.Module):
    def forward(self, input1, input2, bias, bmm_shape, output_shape, output_dtype):
        if len(bmm_shape) == 4:
            batch = bmm_shape[0] * bmm_shape[1]
            m = bmm_shape[-2]
            n = bmm_shape[-1]
        elif len(bmm_shape) == 3:
            batch = bmm_shape[0]
            m = bmm_shape[-2]
            n = bmm_shape[-1]
        else:
            exit(-1)

        if (len(input1.shape) == 4) and (input1.shape[0] == 1):
            input1 = input1.squeeze(0)
        if (len(input2.shape) == 4) and (input2.shape[0] == 1):
            input2 = input2.squeeze(0)

        if input1.shape != (batch, m, input1.numel() // batch // m):
            input1 = input1.contiguous().view(batch, m, -1)
        if input2.shape != (batch, input2.numel() // batch // n, n):
            input2 = input2.contiguous().view(batch, -1, n)

        if input1.dtype != output_dtype:
            input1 = input1.to(output_dtype)
        if input2.dtype != output_dtype:
            input2 = input2.to(output_dtype)

        if bias is not None:
            if bias.dtype != output_dtype:
                bias = bias.to(output_dtype)
            output = torch.ops.torch_npu_triton.triton_bmm(input1, input2) + bias
        else:
            output = torch.ops.torch_npu_triton.triton_bmm(input1, input2)

        if output.shape != output_shape:
            output = output.view(output_shape)
        return output


def _is_bmm(
    node: fx.Node,
):
    match_, input1, input2 = check_bmm_op(node)
    if match_:
        input1 = get_actual_node(node, 0)
        input2 = get_actual_node(node, 1)
        return True, (input1, input2, get_shape(node), get_dtype(node))
    return match_, ()


def check_fused_bmm(node: fx.Node):
    if not check_op(node, "fused_bmm"):
        return False
    return True


def _is_add_bmm(
    node: fx.Node,
):
    if not check_add_op(node):
        return False, ()
    bmm_node = get_actual_node(node, 0)
    add_node = get_actual_node(node, 1)
    if not check_fused_bmm(bmm_node):
        bmm_node = get_actual_node(node, 1)
        add_node = get_actual_node(node, 0)
        if not check_fused_bmm(bmm_node):
            return False, ()
    return True, (bmm_node, add_node, get_shape(node), get_dtype(node))


def replace_node(graph_module, bmm_node, node, target_str):
    with graph_module.graph.inserting_before(node):
        new_node = graph_module.graph.call_module(
            target_str,
            args=(
                bmm_node.args[0],
                bmm_node.args[1],
                bmm_node.args[2],
                bmm_node.args[3],
                get_shape(node),
                get_dtype(node),
            ),
        )
    node.replace_all_uses_with(new_node)
    graph_module.graph.erase_node(node)
    graph_module.graph.erase_node(bmm_node)


def _is_bmm_view(node, target_str):
    if (not check_view(node)) and (not check_copy(node)) and (not check_clone(node)):
        return False, None
    bmm_node = node.args[0]
    if bmm_node.target != target_str:
        return False, None
    if len(bmm_node.users) != 1:
        return False, None
    return True, bmm_node


class FusedBMM(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False

        graph_module.add_submodule("fused_bmm", FusedBAddBMMReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, bmm_param = _is_bmm(node)
            if is_match:
                node1, node2, output_shape, output_dtype = bmm_param
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_bmm",
                        args=(
                            node1,
                            node2,
                            None,
                            output_shape,
                            output_shape,
                            output_dtype,
                        ),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        for node in reversed(graph_module.graph.nodes):
            is_match, bmm_node = _is_bmm_view(node, "fused_bmm")
            if is_match:
                replace_node(graph_module, bmm_node, node, "fused_bmm")

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
