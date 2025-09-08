from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from ...utils.check_ops import check_act_op, get_shape
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


class MMNodeDesc:
    """
    NodeDesc class describes a node in torch.fx graph along with its associated information,
    including input1, input2, bias, and their shapes. It can also store an activation type.
    """

    def __init__(self) -> None:
        # The fx.Node itself (typically an mm or addmm operation)
        self.node: Optional[NodeType] = None
        self.input1: Optional[NodeType] = None
        self.input2: Optional[NodeType] = None
        self.bias: Optional[Union[NodeType, int, float]] = None
        self.input1_shape: Optional[TensorShape] = None
        self.input2_shape: Optional[TensorShape] = None
        self.bias_shape: Optional[TensorShape] = None
        # Activation function string (default "none")
        self.act: str = "none"
        self.input1_ancestors = []
        self.input2_ancestors = []
        self.bias_ancestors = []
        self.extra_match = False

    def set_node(self, node):
        self.node = node

    def set_input1(self, input1):
        self.input1 = input1
        self.input1_shape = get_shape(input1)
        self.input1_ancestors = get_ancestors(self.input1)

    def set_input2(self, input2):
        self.input2 = input2
        self.input2_shape = get_shape(input2)
        self.input2_ancestors = get_ancestors(self.input2)

    def set_bias(self, bias):
        self.bias = bias
        if bias is not None:
            self.bias_shape = get_shape(bias)
            self.bias_ancestors = get_ancestors(self.bias)

    def set_act(self, act: str):
        self.act = act

def get_node_desc(node):
    intpu1 = None
    intpu2 = None
    bias = None
    act = None
    check_args = False
    if node.target in [torch.ops.aten.mm.default, torch.ops.aten.bmm.default]:
        input1 = node.args[0]
        input2 = node.args[1]
    elif node.target == torch.ops.aten.addmm.default:
        bias = node.args[0]
        input1 = node.args[1]
        input2 = node.args[2]
    else:
        # CustomDenseLayer and CustomBatchDenseLayer
        input1 = node.args[0]
        input2 = node.args[1]
        bias = node.args[3]
        # TODO(JYJ):Remove restrictions
        trans_b = node.args[2]
        if trans_b == True:
            return None
        if isinstance(bias, (int, float)):
            return None
        act = node.args[4]

    if get_shape(input1) == None:
        return None
    if get_shape(input2) == None:
        return None
    if bias == True:
        if get_shape(bias) == None:
            return None

    mm_desc = MMNodeDesc()
    mm_desc.set_node(node)
    mm_desc.set_input1(input1)
    mm_desc.set_input2(input2)
    mm_desc.set_bias(bias)
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
