import torch
import operator
from torch import nn, fx
from typing import Union, Tuple

def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def get_input_node(node, idx):
    return node.args[idx]


def get_input_kw_node(node, key):
    if key in node.kwargs:
        return node.kwargs[key]
    else:
        return None


def get_actual_node(node, idx):
    new_node = node.args[idx]
    changed1 = True
    while changed1:
        changed1 = False
        if check_copy(new_node):
            changed1 = True
        if check_view(new_node):
            changed1 = True
        if check_clone(new_node):
            changed1 = True
        if changed1:
            new_node = new_node.args[0]
    return new_node


def check_op(node: fx.Node, target) -> bool:
    return _is_valid_node(node) and node.target == target


def check_sqrt_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.sqrt.default)


def check_rsqrt_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.rsqrt.default)


def check_add_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.add.Tensor)


def check_sub_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.sub.Tensor)


def check_mean_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.mean.dim)


def check_var_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.var.dim)


def check_pow_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.pow.Tensor_Scalar)


def check_mul_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.mul.Tensor)


def check_meta(node: fx.Node) -> bool:
    if node.meta == {}:
        return False
    if "tensor_meta" not in node.meta:
        return False
    return True


def check_meta_2d(node: fx.Node) -> bool:
    if not check_meta(node):
        return False
    if len(node.meta["tensor_meta"].shape) == 2:
        return True
    return False


def get_shape(node: fx.Node):
    return node.meta["tensor_meta"].shape


def get_dtype(node: fx.Node):
    return node.meta["tensor_meta"].dtype


def check_sub_or_add_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None, ()

    if node.target not in [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
    ]:
        return False, None, ()

    is_add = node.target == torch.ops.aten.add.Tensor
    node0 = get_input_node(node, 0)
    node1 = get_input_node(node, 1)
    return True, node0, (node1, is_add)


def check_div_or_mul_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None, ()

    if node.target not in [
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
    ]:
        return False, None, ()

    is_div = node.target == torch.ops.aten.div.Tensor
    node0 = get_input_node(node, 0)
    node1 = get_input_node(node, 1)
    return True, node0, (node1, is_div)


def check_bmm_op(node: fx.Node) -> Tuple[bool, Union[fx.Node, None], Union[fx.Node, None]]:
    if not check_op(node, torch.ops.aten.bmm.default):
        return False, None, None

    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2


def check_mm_op(node: fx.Node) -> Tuple[bool, Union[fx.Node, None], Union[fx.Node, None]]:
    if not check_op(node, torch.ops.aten.mm.default):
        return False, None, None

    arg1 = node.args[0]
    arg2 = node.args[1]
    return True, arg1, arg2


def check_view(node):
    if (not check_op(node, torch.ops.aten._unsafe_view.default)) and (
        not check_op(node, torch.ops.aten.view.default)
    ):
        return False
    return True


def check_softmax_op(node: fx.Node) -> bool:
    if (not check_op(node, torch.ops.aten._safe_softmax.default)) and (
        not check_op(node, torch.ops.aten._softmax.default)
    ):
        return False
    return True


def check_cat_op(node: fx.Node):
    is_cat = check_op(node, torch.ops.aten.cat.default)
    if is_cat:
        if len(node.args) == 1:
            return True, 0
        else:
            return True, node.args[1]
    else:
        return False, 0


def check_slice_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.slice.Tensor)


def check_sum_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.sum.dim_IntList)


def check_stack_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.stack.default)


def check_trans_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.transpose.int)


def check_t_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.t.default)


def check_act_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None
    if node.target == torch.ops.aten.silu.default:
        return True, "silu"
    if node.target == torch.ops.aten.gelu.default:
        return True, "gelu"
    if node.target == torch.ops.aten.relu.default:
        return True, "relu"
    return False, None


def check_copy(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten._to_copy.default)


def check_clone(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.clone.default)


def check_getitem_op(node: fx.node) -> bool:
    return check_op(node, operator.getitem)


def check_mask_fill_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.masked_fill.Scalar)


def check_eq_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.eq.Scalar)


def check_repeat_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.repeat.default)


def check_unsqueeze_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.unsqueeze.default)


def check_norm_op(node: fx.node):
    if not isinstance(node, fx.Node):
        return False, None
    if not (node.op == "call_function" or node.op == "call_module"):
        return False, None
    if node.target == torch.ops.aten.native_layer_norm.default:
        return True, "layer_norm"
    if node.target == "rms_norm_op":
        return True, "rms_norm"
    return False, None


def check_addmm_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[fx.Node, None], Union[fx.Node, None]]:
    if not check_op(node, torch.ops.aten.addmm.default):
        return False, None, None, None
    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    arg3 = get_input_node(node, 2)
    return True, arg1, arg2, arg3
