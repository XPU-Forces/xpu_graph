import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DenseLayer, DenseParams
from xpu_graph.utils import logger

from ..utils.check_ops import check_act_op, check_add_op, check_addmm_op, check_mm_op


def _is_matmul(node: fx.Node) -> Tuple[bool, Optional[DenseParams]]:
    mm_param = DenseParams()
    is_mm, q1, q2 = check_mm_op(node)

    if not is_mm:
        return False, None

    if not mm_param.set_input(q1):
        return False, None

    if not mm_param.set_weight(q2):
        return False, None

    return True, mm_param


def match_mm(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, mm_param = _is_matmul(node)
        if is_match:
            new_node = replace_node(graph_module, node, mm_param, "fused_matmul_replacement")
            changed = True
    return changed


def match_mm_add1(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        if not check_add_op(node):
            continue
        mm_node = node.args[0]
        if not isinstance(mm_node, fx.Node):
            continue
        if mm_node.target != "fused_matmul_replacement":
            continue
        if len(mm_node.users) != 1:
            continue

        mm_param = DenseParams()
        if not mm_param.set_node(mm_node):
            logger.info(f"MatMul Pass: invalid pattern in match_mm_add: {mm_node.name}")
            continue

        bias = node.args[1]

        if not mm_param.set_bias(bias):
            continue
        new_node = replace_node(graph_module, node, mm_param, "fused_matmul_add_replacement")
        assert new_node.args[0] == mm_param.input
        graph_module.graph.erase_node(mm_node)
        changed = True
    return changed


def _is_addmm(node: fx.Node) -> Tuple[bool, Optional[DenseParams]]:
    mm_param = DenseParams()
    match_, bias, input, weight = check_addmm_op(node)
    if not match_:
        return False, None
    if not mm_param.set_input(input):
        return False, None

    if not mm_param.set_weight(weight):
        return False, None

    if not mm_param.set_bias(bias):
        return False, None
    return True, mm_param


def match_mm_add2(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, mm_param = _is_addmm(node)
        if is_match:
            new_node = replace_node(graph_module, node, mm_param, "fused_matmul_add_replacement")
            changed = True
    return changed


def match_mm_act(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_cat, act_str = check_act_op(node)
        if not is_cat:
            continue
        mm_node = node.args[0]
        if (mm_node.target != "fused_matmul_replacement") and (mm_node.target != "fused_matmul_add_replacement"):
            continue
        if len(mm_node.users) != 1:
            continue
        mm_param = DenseParams()
        if not mm_param.set_node(mm_node):
            logger.info(f"MatMul Pass: invalid pattern in match_mm_add: {mm_node.name}")
            continue
        if not mm_param.set_act(act_str):
            continue
        if mm_node.target == "fused_matmul_replacement":
            new_node = replace_node(graph_module, node, mm_param, "fused_matmul_act_replacement")
        elif mm_node.target == "fused_matmul_add_replacement":
            new_node = replace_node(graph_module, node, mm_param, "fused_matmul_add_act_replacement")
        assert new_node.args[0] == mm_param.input
        graph_module.graph.erase_node(mm_node)
        changed = True
    return changed


def replace_node(graph_module, node, mm_param, func_name):
    with graph_module.graph.inserting_before(node):
        new_node = graph_module.graph.call_module(
            func_name,
            args=(
                mm_param.input,
                mm_param.weight,
                mm_param.weight_trans,
                mm_param.bias,
                mm_param.act,
            ),
        )
    node.replace_all_uses_with(new_node)
    graph_module.graph.erase_node(node)
    return new_node


class FusedMatMul(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        # mm
        if not hasattr(graph_module, "fused_matmul_replacement"):
            graph_module.add_submodule("fused_matmul_replacement", DenseLayer())
        is_modified |= match_mm(graph_module)

        return is_modified


class FusedMatMulAdd(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        # mm+bias
        if not hasattr(graph_module, "fused_matmul_add_replacement"):
            graph_module.add_submodule("fused_matmul_add_replacement", DenseLayer())
        is_modified |= match_mm_add1(graph_module)
        is_modified |= match_mm_add2(graph_module)

        return is_modified


class FusedMatMulAct(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        # mm+act
        is_modified = False
        if not hasattr(graph_module, "fused_matmul_act_replacement"):
            graph_module.add_submodule("fused_matmul_act_replacement", DenseLayer())
        # mm+bias+act
        if not hasattr(graph_module, "fused_matmul_add_act_replacement"):
            graph_module.add_submodule("fused_matmul_add_act_replacement", DenseLayer())
        is_modified |= match_mm_act(graph_module)

        return is_modified
