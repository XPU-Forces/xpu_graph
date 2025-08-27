from typing import Optional

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

from .triton_kernel.fused_serial_mm_2dot import fuse_serial_mm_2dot
from .triton_kernel.fused_serial_mm_3dot import fuse_serial_mm_3dot


class FusedSerialMM2DotReplacement(nn.Module):
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


class FusedSerialMM3DotReplacement(nn.Module):
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


def _is_serial_mm_2dot(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if (
        (node.target != "mlu_tmo_fused_matmul_replacement")
        and (node.target != "mlu_tmo_fused_matmul_add_replacement")
        and (node.target != "mlu_tmo_fused_matmul_act_replacement")
        and (node.target != "mlu_tmo_fused_matmul_add_act_replacement")
    ):
        return False, []

    matmul_act_node = node.args[0]
    down_proj_input_shape = node.args[1]
    down_proj_weight = node.args[2]
    down_proj_weight_shape = node.args[3]
    is_transb_down = node.args[4]
    down_proj_bias = node.args[5]
    act_mode_down = node.args[6]

    if (
        (matmul_act_node.target != "mlu_tmo_fused_matmul_replacement")
        and (matmul_act_node.target != "mlu_tmo_fused_matmul_add_replacement")
        and (matmul_act_node.target != "mlu_tmo_fused_matmul_act_replacement")
        and (matmul_act_node.target != "mlu_tmo_fused_matmul_add_act_replacement")
    ):
        return False, []

    if len(matmul_act_node.users) > 1:
        return False, []

    tinyffn_input = matmul_act_node.args[0]
    up_fc_input_shape = matmul_act_node.args[1]
    up_fc_weight = matmul_act_node.args[2]
    up_fc_weight_shape = matmul_act_node.args[3]
    is_transb_up = matmul_act_node.args[4]
    up_fc_bias = matmul_act_node.args[5]
    act_mode_up = matmul_act_node.args[6]

    M, K0 = up_fc_input_shape
    K1, N1 = up_fc_weight_shape
    N2, O2 = down_proj_weight_shape

    if N1 != N2:
        return False, []

    size_of_dtype = 2
    input_dtype = tinyffn_input.meta["val"].dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    if (K1 * N1 + N2 * O2 + N1 + O2) > (512 / size_of_dtype * 1024):
        return False, []

    is_up_bias, is_down_bias = False, False
    if up_fc_bias is not None:
        is_up_bias = True
    if down_proj_bias is not None:
        is_down_bias = True

    is_up_act, is_down_act = True, True
    if act_mode_up == "none":
        is_up_act = False
    elif act_mode_up == "relu":
        is_up_act = True
    else:
        return False, []

    if act_mode_down == "none":
        is_down_act = False
    elif act_mode_down == "relu":
        is_down_act = True
    else:
        return False, []

    if is_transb_up or is_transb_down:
        return False, []

    return True, [
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
    ]


def _is_serial_mm_3dot(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if node.target != "mlu_triton_serial_mm_2dot_replacement":
        return False, []

    matmul_node = node.args[0]
    if (
        (matmul_node.target != "mlu_tmo_fused_matmul_replacement")
        and (matmul_node.target != "mlu_tmo_fused_matmul_add_replacement")
        and (matmul_node.target != "mlu_tmo_fused_matmul_act_replacement")
        and (matmul_node.target != "mlu_tmo_fused_matmul_add_act_replacement")
    ):
        return False, []

    if len(matmul_node.users) > 1:
        return False, []

    M, K1 = matmul_node.args[1]
    K1, N1 = matmul_node.args[3]
    N1, N2 = node.args[1].meta["val"].shape
    N2, N3 = node.args[3].meta["val"].shape

    size_of_dtype = 2
    input_dtype = matmul_node.meta["val"].dtype
    if input_dtype == torch.float32:
        size_of_dtype = 4
    if (K1 * N1 + N1 * N2 + N2 * N3 + N1 + N2 + N3) > (512 / size_of_dtype * 1024):
        return False, []

    first_input = matmul_node.args[0]
    first_weight = matmul_node.args[2]
    first_bias = matmul_node.args[5]
    is_transb_first = matmul_node.args[4]
    act_mode_first = matmul_node.args[6]

    if is_transb_first:
        return False, []

    is_first_bias = False
    if first_bias is not None:
        is_first_bias = True

    is_first_act = True
    if act_mode_first == "none":
        is_first_act = False
    elif act_mode_first == "relu":
        is_first_act = True
    else:
        return False, []

    ffn_params = [
        first_input,
        first_weight,
        first_bias,
    ]
    ffn_params += node.args[1:5]
    ffn_params += [act_mode_first]
    ffn_params += node.args[5:7]
    ffn_params += [is_transb_first]
    ffn_params += node.args[7:9]
    ffn_params += [is_first_bias]
    ffn_params += node.args[9:11]
    ffn_params += [is_first_act]
    ffn_params += node.args[11:]

    return True, ffn_params


class FusedSerialMM2Dot(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        return False
        graph_module.add_submodule("mlu_triton_serial_mm_2dot_replacement", FusedSerialMM2DotReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_serial_mm_2dot(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_serial_mm_2dot_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified


class FusedSerialMM3Dot(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        return False
        graph_module.add_submodule("mlu_triton_serial_mm_3dot_replacement", FusedSerialMM3DotReplacement())
        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_serial_mm_3dot(node)
            if is_match:
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_serial_mm_3dot_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
