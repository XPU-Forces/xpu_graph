import torch
from torch import fx

aten = torch.ops.aten
# WARNING(liuyuan): they just rename the op from a version somehow.
npu_dtype_cast_list = [
    getattr(torch.ops.npu, name).default
    for name in ("npu_dtype_cast", "_npu_dtype_cast")
    if hasattr(torch.ops.npu, name)
]


def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def check_op(node: fx.Node, target) -> bool:
    return _is_valid_node(node) and node.target == target


def check_npu_dtype_cast_op(node: fx.node) -> bool:
    return any(check_op(node, target) for target in npu_dtype_cast_list)


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
    if node.target == torch.ops.npu.npu_rms_norm.default:
        return True, "rms_norm"
    else:
        return False, None
