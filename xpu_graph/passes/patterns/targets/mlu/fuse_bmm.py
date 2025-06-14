from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import check_act_op, check_add_op, check_bmm_op, check_view

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node


class BMMParam:
    def __init__(self) -> None:
        self.input: Optional[NodeType] = None
        self.input_shape: Optional[TensorShape] = None
        self.weight: Optional[NodeType] = None
        self.weight_shape: Optional[TensorShape] = None
        self.trans_b: bool = False
        self.residual: Optional[NodeType] = None
        self.beta: Optional[float] = 0.0
        self.bias: Optional[NodeType] = None
        self.output_dtype: Optional[torch.dtype] = None
        self.act: str = "none"
        self.shape_param: Optional[Tuple[int, ...]] = None
        self.node_name: list = []

    def set_node(self, node):
        if len(node.args) != 11:
            return False
        self.input = node.args[0]
        self.input_shape = node.args[1]
        self.weight = node.args[2]
        self.weight_shape = node.args[3]
        self.trans_b = node.args[4]
        self.residual = node.args[5]
        self.beta = node.args[6]
        self.bias = node.args[7]
        self.output_dtype = node.args[8]
        self.act = node.args[9]
        self.shape_param = node.args[10]
        return True

    def set_input(self, node):
        input_shape = node.meta["val"].shape
        input_dtype = node.meta["val"].dtype
        self.input = node
        self.input_shape = input_shape
        self.output_dtype = input_dtype
        return True

    def set_weight(self, node):
        weight_shape = node.meta["val"].shape
        self.weight = node
        self.weight_shape = weight_shape
        self.trans_b = False
        return True

    def set_bias(self, bias):
        if isinstance(bias, int):
            return False
        if isinstance(bias, float):
            return False

        b1, m1, k1 = self.input_shape
        b2, k2, n2 = self.weight_shape

        bias_shape = bias.meta["val"].shape
        b3, m3, n3 = bias_shape
        if len(bias_shape) < 3:
            return False
        else:
            if b3 != b1 or n3 != n2:
                return False
            elif m3 == 1:
                self.bias = bias
            elif m3 == m1:
                self.residual = bias
                self.beta = 1.0
        return True

    def set_act(self, act_str):
        if act_str in ["gelu", "silu", "none"]:
            self.act = act_str
            return True
        return False

    def set_shape_param(self, shape_param):
        n = self.weight_shape[1] if self.trans_b == False else self.weight_shape[0]
        if len(shape_param) == 3:
            if self.input_shape[0] == shape_param[0] and n == shape_param[2]:
                self.shape_param = None
                return True
        self.shape_param = shape_param
        return True


class FusedBMMReplacement(nn.Module):
    def forward(
        self, inputs, input_shape, weight, weight_shape, trans_b, residual, beta, bias, dtype, act, shape_param
    ):
        import torch_mlu_ops

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = torch_mlu_ops.batch_matmul(
            inputs,
            weight,
            residual,
            1.0,
            beta,
            1.0,
            1.0,
            False,
            trans_b,
            None,
            bias,
            act,
            dtype,
        )
        if shape_param:
            output = output.view(shape_param)
        return output


def _is_bmm(node: NodeType) -> Tuple[bool, Optional[BMMParam]]:
    bmm_param = BMMParam()
    is_bmm, q1, q2 = check_bmm_op(node)

    if not is_bmm:
        return False, None
    if not bmm_param.set_input(q1):
        return False, None
    if not bmm_param.set_weight(q2):
        return False, None

    return True, bmm_param


def replace_node(graph_module, node, bmm_param, func_name):
    with graph_module.graph.inserting_before(node):
        new_node = graph_module.graph.call_module(
            func_name,
            args=(
                bmm_param.input,
                bmm_param.input_shape,
                bmm_param.weight,
                bmm_param.weight_shape,
                bmm_param.trans_b,
                bmm_param.residual,
                bmm_param.beta,
                bmm_param.bias,
                bmm_param.output_dtype,
                bmm_param.act,
                bmm_param.shape_param,
            ),
        )
    node.replace_all_uses_with(new_node)
    graph_module.graph.erase_node(node)
    return new_node


def match_bmm(graph_module):
    graph_module.add_submodule("mlu_tmo_fused_bmm_replacement", FusedBMMReplacement())
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, bmm_param = _is_bmm(node)
        if is_match:
            new_node = replace_node(graph_module, node, bmm_param, "mlu_tmo_fused_bmm_replacement")
            changed = True
    return changed


def match_bmm_add(graph_module):
    graph_module.add_submodule("mlu_tmo_fused_bmm_add_replacement", FusedBMMReplacement())
    changed = False
    for node in reversed(graph_module.graph.nodes):
        if not check_add_op(node):
            continue
        bmm_node = node.args[0]
        if not isinstance(bmm_node, fx.Node):
            continue
        if bmm_node.target != "mlu_tmo_fused_bmm_replacement":
            continue
        if len(bmm_node.users) != 1:
            continue

        bmm_param = BMMParam()
        if not bmm_param.set_node(bmm_node):
            logger.info(f"BMM Pass: invalid pattern in match_bmm_add: {bmm_node.name}")
            continue
        bias = node.args[1]
        if not bmm_param.set_bias(bias):
            continue
        new_node = replace_node(graph_module, node, bmm_param, "mlu_tmo_fused_bmm_add_replacement")
        assert new_node.args[0] == bmm_param.input
        graph_module.graph.erase_node(bmm_node)
        changed = True
    return changed


def match_bmm_act(graph_module):
    # bmm+act
    graph_module.add_submodule("mlu_tmo_fused_bmm_act_replacement", FusedBMMReplacement())
    # bmm+bias+act
    graph_module.add_submodule("mlu_tmo_fused_bmm_add_act_replacement", FusedBMMReplacement())
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_act, act_str = check_act_op(node)
        if not is_act:
            continue
        bmm_node = node.args[0]
        if (bmm_node.target != "mlu_tmo_fused_bmm_replacement") and (
            bmm_node.target != "mlu_tmo_fused_bmm_add_replacement"
        ):
            continue
        if len(bmm_node.users) != 1:
            continue
        bmm_param = BMMParam()
        if not bmm_param.set_node(bmm_node):
            logger.info(f"BMM Pass: invalid pattern in match_bmm_act: {bmm_node.name}")
            continue
        if not bmm_param.set_act(act_str):
            continue
        if bmm_node.target == "mlu_tmo_fused_bmm_replacement":
            new_node = replace_node(graph_module, node, bmm_param, "mlu_tmo_fused_bmm_act_replacement")
        elif bmm_node.target == "mlu_tmo_fused_bmm_add_replacement":
            new_node = replace_node(graph_module, node, bmm_param, "mlu_tmo_fused_bmm_add_act_replacement")
        assert new_node.args[0] == bmm_param.input
        graph_module.graph.erase_node(bmm_node)
        changed = True
    return changed


def match_bmm_view(graph_module, target_str):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        if not check_view(node):
            continue
        bmm_node = node.args[0]
        if bmm_node.target != target_str:
            continue
        if len(bmm_node.users) != 1:
            continue
        bmm_param = BMMParam()
        if not bmm_param.set_node(bmm_node):
            logger.info(f"BMM Pass: invalid pattern in match_bmm_view: {bmm_node.name}")
            continue
        if not bmm_param.set_shape_param(node.args[1]):
            continue
        new_node = replace_node(graph_module, node, bmm_param, bmm_node.target)
        assert new_node.args[0] == bmm_param.input
        graph_module.graph.erase_node(bmm_node)
        changed = True
    return changed


class FusedBMM(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        is_modified |= match_bmm(graph_module)
        is_modified |= match_bmm_view(graph_module, "mlu_tmo_fused_bmm_replacement")

        is_modified |= match_bmm_add(graph_module)
        is_modified |= match_bmm_view(graph_module, "mlu_tmo_fused_bmm_add_replacement")

        is_modified |= match_bmm_act(graph_module)
        is_modified |= match_bmm_view(graph_module, "mlu_tmo_fused_bmm_act_replacement")
        is_modified |= match_bmm_view(graph_module, "mlu_tmo_fused_bmm_add_act_replacement")

        return is_modified
