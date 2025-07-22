import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import (
    BatchDenseLayer,
    BatchDenseParams,
    DenseLayer,
    DenseParams,
    replace_with_module,
)
from xpu_graph.utils import logger

from ..utils.check_ops import (
    check_act_op,
    check_add_op,
    check_addmm_op,
    check_bmm_op,
    check_mm_op,
)


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


def _is_bmm(node: fx.Node) -> Tuple[bool, Optional[BatchDenseParams]]:
    bmm_param = BatchDenseParams()
    is_bmm, q1, q2 = check_bmm_op(node)

    if not is_bmm:
        return False, None
    if not bmm_param.set_input(q1):
        return False, None
    if not bmm_param.set_weight(q2):
        return False, None

    return True, bmm_param


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


def match_mm(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, mm_param = _is_matmul(node)
        if is_match:
            new_node = replace_with_module(graph_module, node, "fused_dense_layer_replacement", mm_param.as_tuple())
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
        if mm_node.target != "fused_dense_layer_replacement" or mm_node.args[3] != None or mm_node.args[4] != "none":
            continue
        if len(mm_node.users) != 1:
            continue

        mm_param = DenseParams()
        if not mm_param.set_node(mm_node):
            logger.debug(f"MatMulAdd Pass: invalid pattern in match_mm_add: {mm_node.name}")
            continue

        bias = node.args[1]

        if not mm_param.set_bias(bias):
            continue
        new_node = replace_with_module(graph_module, node, "fused_dense_layer_replacement", mm_param.as_tuple())
        assert new_node.args[0] == mm_param.input
        graph_module.graph.erase_node(mm_node)
        changed = True
    return changed


def match_mm_add2(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, mm_param = _is_addmm(node)
        if is_match:
            new_node = replace_with_module(graph_module, node, "fused_dense_layer_replacement", mm_param.as_tuple())
            changed = True
    return changed


def match_mm_act(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_cat, act_str = check_act_op(node)
        if not is_cat:
            continue
        mm_node = node.args[0]
        if mm_node.target != "fused_dense_layer_replacement" or mm_node.args[4] != "none":
            continue
        if len(mm_node.users) != 1:
            continue
        mm_param = DenseParams()
        if not mm_param.set_node(mm_node):
            logger.info(f"MatMul Pass: invalid pattern in match_mm_add: {mm_node.name}")
            continue
        if not mm_param.set_act(act_str):
            continue
        new_node = replace_with_module(graph_module, node, "fused_dense_layer_replacement", mm_param.as_tuple())
        assert new_node.args[0] == mm_param.input
        graph_module.graph.erase_node(mm_node)
        changed = True
    return changed


class FusedMatMul(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        # mm
        if not hasattr(graph_module, "fused_dense_layer_replacement"):
            graph_module.add_submodule("fused_dense_layer_replacement", DenseLayer())
        is_modified |= match_mm(graph_module)

        return is_modified


class FusedMatMulAdd(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        # mm+bias
        if not hasattr(graph_module, "fused_dense_layer_replacement"):
            graph_module.add_submodule("fused_dense_layer_replacement", DenseLayer())
        is_modified |= match_mm_add1(graph_module)
        is_modified |= match_mm_add2(graph_module)

        return is_modified


class FusedMatMulAct(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_dense_layer_replacement"):
            graph_module.add_submodule("fused_dense_layer_replacement", DenseLayer())
        # find all mm + act or mm_add + act patterns
        is_modified |= match_mm_act(graph_module)

        return is_modified


def match_bmm(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, bmm_param = _is_bmm(node)
        if is_match:
            new_node = replace_with_module(
                graph_module, node, "fused_batch_dense_layer_replacement", bmm_param.as_tuple()
            )
            changed = True
    return changed


def match_bmm_add(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        if not check_add_op(node):
            continue
        bmm_node = node.args[0]
        if not isinstance(bmm_node, fx.Node):
            continue
        if (
            bmm_node.target != "fused_batch_dense_layer_replacement"
            or bmm_node.args[3] != None
            or bmm_node.args[4] != None
            or bmm_node.args[5] != "none"
        ):
            continue
        if len(bmm_node.users) != 1:
            continue

        bmm_param = BatchDenseParams()
        if not bmm_param.set_node(bmm_node):
            logger.debug(f"BatchDenseLayer Pass: invalid pattern in match_bmm_add: {bmm_node.name}")
            continue
        bias = node.args[1]
        if not bmm_param.set_bias(bias):
            continue
        new_node = replace_with_module(graph_module, node, "fused_batch_dense_layer_replacement", bmm_param.as_tuple())
        assert new_node.args[0] == bmm_param.input
        graph_module.graph.erase_node(bmm_node)
        changed = True
    return changed


def match_bmm_act(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_act, act_str = check_act_op(node)
        if not is_act:
            continue
        bmm_node = node.args[0]
        if bmm_node.target != "fused_batch_dense_layer_replacement" or bmm_node.args[5] != "none":
            continue
        if len(bmm_node.users) != 1:
            continue
        bmm_param = BatchDenseParams()
        if not bmm_param.set_node(bmm_node):
            logger.info(f"BatchDenseLayer Pass: invalid pattern in match_bmm_act: {bmm_node.name}")
            continue
        if not bmm_param.set_act(act_str):
            continue
        new_node = replace_with_module(graph_module, node, "fused_batch_dense_layer_replacement", bmm_param.as_tuple())
        assert new_node.args[0] == bmm_param.input
        graph_module.graph.erase_node(bmm_node)
        changed = True
    return changed


class FusedBMM(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_batch_dense_layer_replacement"):
            graph_module.add_submodule("fused_batch_dense_layer_replacement", BatchDenseLayer())
        # find all bmm patterns
        is_modified |= match_bmm(graph_module)
        return is_modified


class FusedBMMAdd(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_batch_dense_layer_replacement"):
            graph_module.add_submodule("fused_batch_dense_layer_replacement", BatchDenseLayer())
        # find all bmm + add patterns
        is_modified |= match_bmm_add(graph_module)
        return is_modified


class FusedBMMAct(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_batch_dense_layer_replacement"):
            graph_module.add_submodule("fused_batch_dense_layer_replacement", BatchDenseLayer())
        # find all bmm + act / bmm_add + act patterns
        is_modified |= match_bmm_act(graph_module)
        return is_modified
