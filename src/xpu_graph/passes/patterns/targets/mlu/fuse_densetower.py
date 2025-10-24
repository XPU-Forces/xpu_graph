from typing import Optional

import torch
from torch import fx
from torch.fx import map_arg

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from .triton_kernel.fused_serial_mm_2dot import fuse_serial_mm_2dot
from .triton_kernel.fused_serial_mm_3dot import fuse_serial_mm_3dot


class FusedDenseTower2Replacement(torch.nn.Module):
    def forward(
        self,
        tinyffn_input,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        act_mode_up,
        act_mode_down,
        is_transb_up,
        is_transb_down,
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    ):
        if not tinyffn_input.is_contiguous():
            tinyffn_input = tinyffn_input.contiguous()
        if not up_fc_weight.is_contiguous():
            up_fc_weight = up_fc_weight.contiguous()
        if not down_proj_weight.is_contiguous():
            down_proj_weight = down_proj_weight.contiguous()
        output = fuse_serial_mm_2dot(
            tinyffn_input,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            is_up_bias,
            is_down_bias,
            is_up_act,
            is_down_act,
        )
        return output


def can_fuse_dense_tower2(
    tinyffn_input,
    up_fc_weight,
    up_fc_bias,
    down_proj_weight,
    down_proj_bias,
    act_mode_up,
    act_mode_down,
    is_transb_up,
    is_transb_down,
    is_up_bias,
    is_down_bias,
    is_up_act,
    is_down_act,
):
    K1, N1 = up_fc_weight.shape
    N2, O2 = down_proj_weight.shape

    if N1 != N2:
        return False, []

    size_of_dtype = 2
    input_dtype = tinyffn_input.dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    # Note: all weights should accommodate the wram size (512KB)
    return (K1 * N1 + N2 * O2 + N1 + O2) <= (512 / size_of_dtype * 1024)


class FusedDenseTower3Replacement(torch.nn.Module):
    def forward(
        self,
        first_input,
        first_weight,
        first_bias,
        up_fc_weight,
        up_fc_bias,
        down_proj_weight,
        down_proj_bias,
        act_mode_first,
        act_mode_up,
        act_mode_down,
        is_transb_first,
        is_transb_up,
        is_transb_down,
        is_first_bias,
        is_up_bias,
        is_down_bias,
        is_first_act,
        is_up_act,
        is_down_act,
    ):
        if not first_input.is_contiguous():
            first_input = first_input.contiguous()
        if not first_weight.is_contiguous():
            first_weight = first_weight.contiguous()
        if not up_fc_weight.is_contiguous():
            up_fc_weight = up_fc_weight.contiguous()
        if not down_proj_weight.is_contiguous():
            down_proj_weight = down_proj_weight.contiguous()
        output = fuse_serial_mm_3dot(
            first_input,
            first_weight,
            first_bias,
            up_fc_weight,
            up_fc_bias,
            down_proj_weight,
            down_proj_bias,
            is_first_bias,
            is_up_bias,
            is_down_bias,
            is_first_act,
            is_up_act,
            is_down_act,
        )
        return output


def can_fuse_dense_tower3(
    first_input,
    first_weight,
    first_bias,
    up_fc_weight,
    up_fc_bias,
    down_proj_weight,
    down_proj_bias,
    act_mode_first,
    act_mode_up,
    act_mode_down,
    is_transb_first,
    is_transb_up,
    is_transb_down,
    is_first_bias,
    is_up_bias,
    is_down_bias,
    is_first_act,
    is_up_act,
    is_down_act,
):
    K1, N1 = first_weight.shape
    N1, N2 = up_fc_weight.shape
    N2, N3 = down_proj_weight.shape

    size_of_dtype = 2
    input_dtype = first_input.dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    # Note: all weights should accommodate the wram size (512KB)
    return (K1 * N1 + N1 * N2 + N2 * N3 + N1 + N2 + N3) <= (512 / size_of_dtype * 1024)


def _is_dense_layer_node(node):
    return isinstance(node, fx.Node) and node.op == "call_module" and node.target == "custom_dense_layer_replacement"


def _is_serial_mm_2dot(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if not _is_dense_layer_node(node):
        return False, []

    inputs1, weight1, weight_trans1, bias1, act1 = node.args

    dense_result0 = inputs1
    if not _is_dense_layer_node(dense_result0):
        return False, []

    if len(dense_result0.users) > 1:
        return False, []

    inputs0, weight0, weight_trans0, bias0, act0 = dense_result0.args

    is_up_bias, is_down_bias = False, False
    if bias0 is not None:
        is_up_bias = True
    if bias1 is not None:
        is_down_bias = True

    is_up_act, is_down_act = True, True
    if act0 == "none":
        is_up_act = False
    elif act0 == "relu":
        is_up_act = True
    else:
        return False, []

    if act1 == "none":
        is_down_act = False
    elif act1 == "relu":
        is_down_act = True
    else:
        return False, []

    if weight_trans0 or weight_trans1:
        return False, []

    return True, [
        inputs0,
        weight0,
        bias0,
        weight1,
        bias1,
        act0,
        act1,
        weight_trans0,
        weight_trans1,
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    ]


def _is_serial_mm_3dot(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if node.target != "fused_dense_tower_2_replacement":
        return False, []

    matmul_node = node.args[0]
    if not _is_dense_layer_node(matmul_node):
        return False, []

    if len(matmul_node.users) > 1:
        return False, []

    inputs0, weight0, weight_trans0, bias0, act0 = matmul_node.args

    if weight_trans0:
        return False, []

    is_first_bias = False
    if bias0 is not None:
        is_first_bias = True

    is_first_act = True
    if act0 == "none":
        is_first_act = False
    elif act0 == "relu":
        is_first_act = True
    else:
        return False, []

    ffn_params = [
        inputs0,
        weight0,
        bias0,
    ]
    ffn_params += node.args[1:5]
    ffn_params += [act0]
    ffn_params += node.args[5:7]
    ffn_params += [weight_trans0]
    ffn_params += node.args[7:9]
    ffn_params += [is_first_bias]
    ffn_params += node.args[9:11]
    ffn_params += [is_first_act]
    ffn_params += node.args[11:]

    return True, ffn_params


class FusedDenseTower2(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_dense_tower_2_replacement"):
            graph_module.add_submodule("fused_dense_tower_2_replacement", FusedDenseTower2Replacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_serial_mm_2dot(node)
            if is_match and can_fuse_dense_tower2(*map_arg(tinyffn_param, lambda x: x.meta["val"])):
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_dense_tower_2_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified


class FusedDenseTower3(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_dense_tower_3_replacement"):
            graph_module.add_submodule("fused_dense_tower_3_replacement", FusedDenseTower3Replacement())

        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_serial_mm_3dot(node)
            if is_match and can_fuse_dense_tower3(*map_arg(tinyffn_param, lambda x: x.meta["val"])):
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_dense_tower_3_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
