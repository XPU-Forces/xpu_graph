from typing import Optional, Tuple, Union

import torch
import torch.fx as fx

from ...utils.check_ops import (  # get_shape,
    check_act_op,
    check_permute_op,
    check_t_op,
    check_trans_op,
    get_shape,
)
from ...utils.combo_utils import get_ancestors


def has_mm_dependency(a, b):
    all_a = a.input1_ancestors + a.input2_ancestors + a.bias_ancestors
    all_b = b.input1_ancestors + b.input2_ancestors + b.bias_ancestors
    return (
        a.input1 in all_b
        or a.input2 in all_b
        or (a.bias in all_b if a.bias else False)
        or b.input1 in all_a
        or b.input2 in all_a
        or (b.bias in all_a if b.bias else False)
    )


def check_trans_input(node, final_trans):
    while True:
        is_trans, node = check_input(node)
        if is_trans:
            final_trans = not final_trans
        else:
            break
    return final_trans, node


def check_input(node):
    weight_trans = False
    if check_trans_op(node):
        trans_param = (node.args[1], node.args[2])
        if trans_param in [(0, 1), (1, 0), (-2, -1), (-1, -2)]:
            weight_trans = True
            node = node.args[0]
    elif check_permute_op(node):
        trans_param = node.args[1]
        if trans_param in [[0, 1], [1, 0], [-2, -1], [-1, -2]]:
            weight_trans = True
            node = node.args[0]
    elif check_t_op(node):
        weight_trans = True
        node = node.args[0]
    return weight_trans, node


class MMNodeDesc:
    """
    NodeDesc class describes a node in torch.fx graph along with its associated information,
    including input1, input2, bias, and their shapes. It can also store an activation type.
    """

    def __init__(self) -> None:
        # The fx.Node itself (typically an mm or addmm operation)
        self.node: Optional[fx.Node] = None
        self.input1: Optional[fx.Node] = None
        self.input2: Optional[fx.Node] = None
        self.bias: Optional[Union[fx.Node, int, float]] = None
        self.input1_shape: Optional[torch.Size] = None
        self.input2_shape: Optional[torch.Size] = None
        self.bias_shape: Optional[torch.Size] = None
        # Activation function string (default "none")
        self.act: str = "none"
        self.input1_ancestors = []
        self.input2_ancestors = []
        self.bias_ancestors = []
        self.extra_match = False
        self.input1_trans = False
        self.input2_trans = False

    def set_node(self, node):
        self.node = node

    def set_input1(self, input1, final_trans=False):
        self.input1_trans, self.input1 = check_trans_input(input1, final_trans)
        self.input1_shape = get_shape(self.input1)
        if self.input1_shape == None:
            return False
        self.input1_ancestors = get_ancestors(self.input1)
        return True

    def set_input2(self, input2, final_trans=False):
        self.input2_trans, self.input2 = check_trans_input(input2, final_trans)
        self.input2_shape = get_shape(self.input2)
        if self.input2_shape == None:
            return False
        if len(self.input2_shape) == 1:
            # Note: skip dot-product cases
            return False
        self.input2_ancestors = get_ancestors(self.input2)
        return True

    def set_bias(self, bias):
        self.bias = bias
        if bias is not None:
            self.bias_shape = get_shape(bias)
            if self.bias_shape == None:
                return False
            self.bias_ancestors = get_ancestors(self.bias)
        return True

    def set_act(self, act: str):
        self.act = act


def get_node_desc(node):
    intpu1 = None
    intpu2 = None
    bias = None
    act = None
    check_args = False
    trans_b = False
    if node.target in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default, torch.ops.aten.bmm.default):
        input1 = node.args[0]
        input2 = node.args[1]
    elif node.target == torch.ops.aten.addmm.default:
        bias = node.args[0]
        input1 = node.args[1]
        input2 = node.args[2]
    elif node.target == torch.ops.aten.linear.default:
        input1 = node.args[0]
        input2 = node.args[1]
        if len(node.args) == 3:
            bias = node.args[2]
        trans_b = True
    else:
        # CustomDenseLayer and CustomBatchDenseLayer
        input1 = node.args[0]
        input2 = node.args[1]
        bias = node.args[3]
        # TODO(JYJ):Remove restrictions
        trans_b = node.args[2]
        act = node.args[4]

    if isinstance(bias, (int, float)):
        return None

    mm_desc = MMNodeDesc()
    mm_desc.set_node(node)
    if not mm_desc.set_input1(input1):
        return None
    if not mm_desc.set_input2(input2, trans_b):
        return None
    if not mm_desc.set_bias(bias):
        return None
    mm_desc.set_act(act)

    # extra match
    # for addmm + relu
    if act == None:
        if len(node.users) == 1:
            next_node = next(iter(node.users))
            is_act, act_str = check_act_op(next_node)
            if is_act:
                mm_desc.extra_match = True
                mm_desc.set_act(act_str)

    return mm_desc
