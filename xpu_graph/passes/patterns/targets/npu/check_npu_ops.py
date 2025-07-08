import torch
from torch import fx

aten = torch.ops.aten


def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def check_op(node: fx.Node, target) -> bool:
    return _is_valid_node(node) and node.target == target


def check_npu_dtype_cast_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.npu.npu_dtype_cast.default)


def check_npu_typecast_op(node: fx.Node) -> bool:
    if check_npu_dtype_cast_op(node):
        return True, node.args[1]
    else:
        return False, None


def check_npu_norm_op(node: fx.node):
    if not isinstance(node, fx.Node):
        return False, None
    if not (node.op == "call_function" or node.op == "call_module"):
        return False, None
    if "npu_rms_norm" in node.name:
        return True, "rms_norm"
    else:
        return False, None
