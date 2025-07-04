import operator
from typing import Any, Tuple, Union

import torch
from torch import fx, nn
from torch.fx.operator_schemas import normalize_function, normalize_module

aten = torch.ops.aten


def _is_valid_node(node: fx.Node) -> bool:
    return isinstance(node, fx.Node) and node.op == "call_function"


def get_input_node(node, idx):
    if node.op == "call_function":
        args_kwargs = normalize_function(node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=False)
    elif node.op == "call_module":
        root = node.graph.owning_module
        args_kwargs = normalize_module(
            root,
            node.target,
            node.args,
            node.kwargs,
            normalize_to_only_use_kwargs=False,
        )
    else:
        args_kwargs = None
    if args_kwargs is not None:
        args, _ = args_kwargs
        if idx < len(args) and idx >= -len(args):
            return args[idx]
    return None


def get_input_kw_node(node, key):
    from torch.fx.operator_schemas import normalize_function, normalize_module

    if node.op == "call_function":
        args_kwargs = normalize_function(node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True)
    elif node.op == "call_module":
        root = node.graph.owning_module
        args_kwargs = normalize_module(root, node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True)
    else:
        args_kwargs = None
    if args_kwargs is not None:
        _, kwargs = args_kwargs
        if key in kwargs:
            return kwargs[key]
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
    return check_op(node, aten.sqrt.default)


def check_rsqrt_op(node: fx.Node) -> bool:
    return check_op(node, aten.rsqrt.default)


def check_add_op(node: fx.Node) -> bool:
    return check_op(node, aten.add.Tensor)


def check_sub_op(node: fx.Node) -> bool:
    return check_op(node, aten.sub.Tensor)


def check_div_op(node: fx.Node) -> bool:
    return check_op(node, torch.ops.aten.div.Tensor)


def check_mean_op(node: fx.Node) -> bool:
    return check_op(node, aten.mean.dim)


def check_var_op(node: fx.Node) -> bool:
    return check_op(node, aten.var.dim)


def check_pow_op(node: fx.Node) -> bool:
    return check_op(node, aten.pow.Tensor_Scalar)


def check_mul_op(node: fx.Node) -> bool:
    return check_op(node, aten.mul.Tensor)


def check_meta(node: fx.Node) -> bool:
    if node.meta == {}:
        return False
    if "val" not in node.meta:
        return False
    return True


def check_meta_2d(node: fx.Node) -> bool:
    if not check_meta(node):
        return False
    if len(node.meta["val"].shape) == 2:
        return True
    return False


def get_shape(node: fx.Node):
    return node.meta["val"].shape


def get_dtype(node: fx.Node):
    return node.meta["val"].dtype


def check_sub_or_add_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None, ()

    if node.target not in [
        aten.add.Tensor,
        aten.sub.Tensor,
    ]:
        return False, None, ()

    is_add = node.target == aten.add.Tensor
    node0 = get_input_node(node, 0)
    node1 = get_input_node(node, 1)
    return True, node0, (node1, is_add)


def check_div_or_mul_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None, ()

    if node.target not in [
        aten.div.Tensor,
        aten.mul.Tensor,
    ]:
        return False, None, ()

    is_div = node.target == aten.div.Tensor
    node0 = get_input_node(node, 0)
    node1 = get_input_node(node, 1)
    return True, node0, (node1, is_div)


def check_bmm_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[fx.Node, None]]:
    if not check_op(node, aten.bmm.default):
        return False, None, None

    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    return True, arg1, arg2


def check_mm_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[fx.Node, None]]:
    if check_op(node, aten.mm.default) or (check_op(node, aten.matmul.default) and check_meta_2d(node)):
        arg1 = node.args[0]
        arg2 = node.args[1]
        return True, arg1, arg2
    return False, None, None


def check_view(node):
    if (not check_op(node, aten._unsafe_view.default)) and (not check_op(node, aten.view.default)):
        return False
    return True


def check_softmax_op(node: fx.Node) -> bool:
    if (not check_op(node, aten._safe_softmax.default)) and (not check_op(node, aten._softmax.default)):
        return False
    return True


def check_cat_op(node: fx.Node):
    is_cat = check_op(node, aten.cat.default)
    if is_cat:
        if len(node.args) == 1:
            return True, 0
        else:
            return True, node.args[1]
    else:
        return False, 0


def check_slice_op(node: fx.Node) -> bool:
    return check_op(node, aten.slice.Tensor)


def check_sum_op(node: fx.Node) -> bool:
    return check_op(node, aten.sum.dim_IntList)


def check_stack_op(node: fx.Node) -> bool:
    return check_op(node, aten.stack.default)


def check_trans_op(node: fx.Node) -> bool:
    return check_op(node, aten.transpose.int)


def check_t_op(node: fx.Node) -> bool:
    return check_op(node, aten.t.default)


def check_act_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[Tuple[Union[fx.Node, None], bool], None]]:
    if not _is_valid_node(node):
        return False, None
    if node.target == aten.silu.default:
        return True, "silu"
    if node.target == aten.gelu.default:
        return True, "gelu"
    if node.target == aten.relu.default:
        return True, "relu"
    if node.target == aten.sigmoid.default:
        return True, "sigmoid"
    return False, None


def check_copy(node: fx.Node) -> bool:
    return check_op(node, aten._to_copy.default)


def check_typecast_op(node: fx.Node) -> bool:
    if (
        check_copy(node)
        and node.meta["tensor_meta"].memory_format == torch.contiguous_format
        and "dtype" in node.kwargs
    ):
        return True, node.kwargs["dtype"]
    else:
        return False, None


def check_clone(node: fx.Node) -> bool:
    return check_op(node, aten.clone.default)


def check_getitem_op(node: fx.node) -> bool:
    return check_op(node, operator.getitem)


def check_mask_fill_op(node: fx.node) -> bool:
    return check_op(node, aten.masked_fill.Scalar)


def check_eq_op(node: fx.node) -> bool:
    return check_op(node, aten.eq.Scalar)


def check_repeat_op(node: fx.node) -> bool:
    return check_op(node, aten.repeat.default)


def check_unsqueeze_op(node: fx.node) -> bool:
    return check_op(node, aten.unsqueeze.default)


def check_norm_op(node: fx.node):
    if not isinstance(node, fx.Node):
        return False, None
    if not (node.op == "call_function" or node.op == "call_module"):
        return False, None
    if node.target == aten.native_layer_norm.default:
        return True, "layer_norm"
    if node.target == "rms_norm_op":
        return True, "rms_norm"
    return False, None


def check_addmm_op(
    node: fx.Node,
) -> Tuple[bool, Union[fx.Node, None], Union[fx.Node, None], Union[fx.Node, None]]:
    if not check_op(node, aten.addmm.default):
        return False, None, None, None
    arg1 = get_input_node(node, 0)
    arg2 = get_input_node(node, 1)
    arg3 = get_input_node(node, 2)
    return True, arg1, arg2, arg3


def check_where_op(node: fx.node) -> bool:
    return check_op(node, aten.where.self)


def check_zeros_op(node: fx.node) -> bool:
    return check_op(node, aten.zeros.default)


def check_squeeze_op(node: fx.node) -> bool:
    return (
        check_op(node, torch.ops.aten.squeeze.default)
        or check_op(node, torch.ops.aten.squeeze.dim)
        or check_op(node, torch.ops.aten.squeeze.dims)
    )


def check_expand_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.expand.default)


def check_rsub_scalar_op(node: fx.node) -> bool:
    return check_op(node, torch.ops.aten.rsub.Scalar)


def is_zero_like(node: Any) -> bool:
    if type(node) in (int, float):
        return node == 0
    elif isinstance(node, fx.Node):
        return node.target in (
            aten.zeros.default,
            aten.zeros_like.default,
        )
    return False


def is_one_like(node: Any) -> bool:
    if type(node) in (int, float):
        return node == 1
    elif isinstance(node, fx.Node):
        return node.target in (
            aten.ones.default,
            aten.ones_like.default,
        )
    return False
