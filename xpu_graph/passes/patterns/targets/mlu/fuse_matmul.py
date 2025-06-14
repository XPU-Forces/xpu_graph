import os
from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import (
    check_act_op,
    check_add_op,
    check_addmm_op,
    check_bmm_op,
    check_mm_op,
    check_t_op,
    check_trans_op,
    check_view,
)

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node


class MMParam:
    def __init__(self) -> None:
        self.input: Optional[NodeType] = None
        self.input_shape: Optional[TensorShape] = None
        self.weight1: Optional[NodeType] = None
        self.weight1_trans: bool = False
        self.weight1_shape: Optional[TensorShape] = None
        self.weight2: Optional[NodeType] = None
        self.weight2_trans: bool = False
        self.weight2_shape: Optional[TensorShape] = None
        self.bias1: Optional[Union[NodeType, int, float]] = None
        self.bias2: Optional[Union[NodeType, int, float]] = None
        self.act: str = "none"
        self.node_name: list = []

    def set_node(self, node):
        if len(node.args) != 7:
            return False
        self.input = node.args[0]
        self.input_shape = node.args[1]
        self.weight1 = node.args[2]
        self.weight1_shape = node.args[3]
        self.weight1_trans = node.args[4]

        if node.args[5] is not None:
            if not self.set_bias1(node.args[5]):
                return False

        if not self.set_act(node.args[6]):
            return False

        return True

    def set_weight1(self, node):
        if check_trans_op(node):
            trans_param = (node.args[1], node.args[2])
            if trans_param in [(0, 1), (1, 0)]:
                self.weight1_trans = True
                node = node.args[0]
            else:
                return False
        elif check_t_op(node):
            self.weight1_trans = True
            node = node.args[0]
        weight1_shape = node.meta["val"].shape
        if len(weight1_shape) != 2:
            logger.warning(f"MatMul pass: Unsupported weight dim {weight1_shape}")
            return False
        self.weight1 = node
        self.weight1_shape = weight1_shape

        if self.input:
            return self.check_shape()
        return True

    def set_input(self, node):
        input_shape = node.meta["val"].shape
        if len(input_shape) != 2:
            logger.warning(f"MatMul pass: Unsupported input dim {input_shape}")
            return False
        self.input_shape = input_shape
        self.input = node

        if self.weight1:
            return self.check_shape()
        return True

    def check_shape(self):
        if self.input is None:
            return False
        if self.weight1 is None:
            return False

        m1, k1 = self.input_shape
        if self.weight1_trans == False:
            k2, n2 = self.weight1_shape
        else:
            n2, k2 = self.weight1_shape
        if k1 != k2:
            logger.warning(
                f"MatMul pass: Unsupported dim input_shape: {self.input_shape}, weight_shape: {self.weight1_shape}"
            )
            return False
        return True

    def set_bias1(self, bias):
        if isinstance(bias, int):
            self.bias1 = bias
            return True
        if isinstance(bias, float):
            self.bias1 = bias
            return True

        m1, k1 = self.input_shape
        if self.weight1_trans == False:
            k2, n2 = self.weight1_shape
        else:
            n2, k2 = self.weight1_shape

        bias_shape = bias.meta["val"].shape
        if len(bias_shape) == 1:
            if (bias_shape != torch.Size([n2])) and (bias_shape != torch.Size([1])):
                return False
        elif len(bias_shape) == 2:
            m3, n3 = bias_shape
            if n2 != n3:
                return False
            if (m1 != m3) and (m3 != 1):
                return False
        self.bias1 = bias
        return True

    def set_act(self, act_str):
        if act_str in ["gelu", "relu", "silu", "sigmoid", "none"]:
            self.act = act_str
            return True
        return False


class FusedMatMulReplacement(nn.Module):
    def __init__(self, fast_act):
        super().__init__()
        self.fast_act = fast_act

    def forward(self, inputs, input_shape, weight, weight_shape, trans_b, bias, act):
        import torch_mlu_ops

        # TODO(jyj): waiting for tmo version update
        tmp_act = act
        if act == "sigmoid":
            tmp_act = "none"

        # input last dim must be contiguous.
        if inputs.stride()[-1] != 1:
            inputs = inputs.contiguous()

        if bias != None:
            if isinstance(bias, int):
                dim = weight.shape[1] if trans_b == False else weight.shape[0]
                bias = torch.tensor([bias] * dim, device=inputs.device, dtype=inputs.dtype)
            bias_shape = bias.shape
            if (len(bias_shape) == 2) & (bias_shape[0] == 1):
                bias = bias.view(-1)
                bias_shape = bias.shape
            if len(bias_shape) == 1:
                output = torch_mlu_ops.matmul(
                    inputs,
                    weight,
                    bias,
                    None,
                    tmp_act,
                    1.0,
                    0.0,
                    self.fast_act,
                    False,
                    trans_b=trans_b,
                )
                if act == "sigmoid":
                    return torch.sigmoid(output)
                return output

        # bias 2d or None
        output = torch_mlu_ops.matmul(
            inputs,
            weight,
            None,
            bias,
            tmp_act,
            1.0,
            0.0 if bias is None else 1.0,
            self.fast_act,
            False,
            trans_b=trans_b,
        )
        if act == "sigmoid":
            return torch.sigmoid(output)
        return output


def _is_matmul(node: NodeType) -> Tuple[bool, Optional[MMParam]]:
    mm_param = MMParam()
    is_mm, q1, q2 = check_mm_op(node)

    if not is_mm:
        return False, None

    if not mm_param.set_input(q1):
        return False, None

    if not mm_param.set_weight1(q2):
        return False, None

    return True, mm_param


def replace_node(graph_module, node, mm_param, func_name):
    with graph_module.graph.inserting_before(node):
        new_node = graph_module.graph.call_module(
            func_name,
            args=(
                mm_param.input,
                mm_param.input_shape,
                mm_param.weight1,
                mm_param.weight1_shape,
                mm_param.weight1_trans,
                mm_param.bias1,
                mm_param.act,
            ),
        )
    node.replace_all_uses_with(new_node)
    graph_module.graph.erase_node(node)
    return new_node


def match_mm(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, mm_param = _is_matmul(node)
        if is_match:
            new_node = replace_node(graph_module, node, mm_param, "mlu_tmo_fused_matmul_replacement")
            changed = True
    return changed


def swap_view_between(graph_module, func1, func2):
    for node in reversed(graph_module.graph.nodes):
        if not func1(node):
            continue
        view_node = node.args[0]
        if not check_view(view_node):
            continue
        mm_node = view_node.args[0]
        if not func2(mm_node):
            continue
        if len(mm_node.users) != 1:
            continue
        if len(view_node.users) != 1:
            continue

        with graph_module.graph.inserting_before(view_node):
            new_node = graph_module.graph.call_function(
                node.target,
                args=(mm_node,),
                kwargs=node.kwargs,
            )
        view_node.args = (new_node,) + view_node.args[1:]
        node.replace_all_uses_with(view_node)
        graph_module.graph.erase_node(node)


def match_mm_add1(graph_module):
    changed = False
    # swap view
    swap_view_between(
        graph_module,
        check_add_op,
        lambda a: a.target != "mlu_tmo_fused_matmul_replacement",
    )

    for node in reversed(graph_module.graph.nodes):
        if not check_add_op(node):
            continue
        mm_node = node.args[0]
        if not isinstance(mm_node, fx.Node):
            continue
        if mm_node.target != "mlu_tmo_fused_matmul_replacement":
            continue
        if len(mm_node.users) != 1:
            continue

        mm_param = MMParam()
        if not mm_param.set_node(mm_node):
            logger.info(f"MatMul Pass: invalid pattern in match_mm_add: {mm_node.name}")
            continue
        bias = node.args[1]
        if not mm_param.set_bias1(bias):
            continue
        new_node = replace_node(graph_module, node, mm_param, "mlu_tmo_fused_matmul_add_replacement")
        assert new_node.args[0] == mm_param.input
        graph_module.graph.erase_node(mm_node)
        changed = True
    return changed


def _is_addmm(node: NodeType) -> Tuple[bool, Optional[MMParam]]:
    mm_param = MMParam()
    match_, input1, input2, input3 = check_addmm_op(node)
    if not match_:
        return False, None
    if not mm_param.set_input(input2):
        return False, None

    if not mm_param.set_weight1(input3):
        return False, None

    if not mm_param.set_bias1(input1):
        return False, None
    return True, mm_param


def match_mm_add2(graph_module):
    changed = False
    for node in reversed(graph_module.graph.nodes):
        is_match, mm_param = _is_addmm(node)
        if is_match:
            new_node = replace_node(graph_module, node, mm_param, "mlu_tmo_fused_matmul_add_replacement")
            changed = True
    return changed


def match_mm_act(graph_module):
    changed = False
    swap_view_between(
        graph_module,
        lambda a: check_act_op(a)[0],
        lambda a: a.target != "mlu_tmo_fused_matmul_replacement",
    )
    swap_view_between(
        graph_module,
        lambda a: check_act_op(a)[0],
        lambda a: a.target != "mlu_tmo_fused_matmul_add_replacement",
    )

    for node in reversed(graph_module.graph.nodes):
        is_cat, act_str = check_act_op(node)
        if not is_cat:
            continue
        mm_node = node.args[0]
        if (mm_node.target != "mlu_tmo_fused_matmul_replacement") and (
            mm_node.target != "mlu_tmo_fused_matmul_add_replacement"
        ):
            continue
        if len(mm_node.users) != 1:
            continue
        mm_param = MMParam()
        if not mm_param.set_node(mm_node):
            logger.info(f"MatMul Pass: invalid pattern in match_mm_add: {mm_node.name}")
            continue
        if not mm_param.set_act(act_str):
            continue
        if mm_node.target == "mlu_tmo_fused_matmul_replacement":
            new_node = replace_node(graph_module, node, mm_param, "mlu_tmo_fused_matmul_act_replacement")
        elif mm_node.target == "mlu_tmo_fused_matmul_add_replacement":
            new_node = replace_node(graph_module, node, mm_param, "mlu_tmo_fused_matmul_add_act_replacement")
        assert new_node.args[0] == mm_param.input
        graph_module.graph.erase_node(mm_node)
        changed = True
    return changed


class FusedMatMul(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        fast_act = True if self._opt_level == OptLevel.level3 else False
        graph_module.add_submodule("mlu_tmo_fused_matmul_replacement", FusedMatMulReplacement(fast_act=fast_act))
        is_modified |= match_mm(graph_module)

        return is_modified


class FusedMatMulAdd(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        fast_act = True if self._opt_level == OptLevel.level3 else False
        graph_module.add_submodule("mlu_tmo_fused_matmul_add_replacement", FusedMatMulReplacement(fast_act=fast_act))
        is_modified |= match_mm_add1(graph_module)
        is_modified |= match_mm_add2(graph_module)

        return is_modified


class FusedMatMulAct(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        # mm+act
        is_modified = False
        fast_act = True if self._opt_level == OptLevel.level3 else False
        graph_module.add_submodule("mlu_tmo_fused_matmul_act_replacement", FusedMatMulReplacement(fast_act=fast_act))
        # mm+bias+act
        graph_module.add_submodule(
            "mlu_tmo_fused_matmul_add_act_replacement", FusedMatMulReplacement(fast_act=fast_act)
        )
        is_modified |= match_mm_act(graph_module)

        return is_modified
