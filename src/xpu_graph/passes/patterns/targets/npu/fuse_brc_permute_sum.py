from typing import List, Optional, Tuple

import torch
import torch.fx as fx
import torch_npu
from torch import fx, nn
from torch.fx.node import Node

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ...utils.check_ops import (
    check_add_op,
    check_expand_op,
    check_mask_fill_op,
    check_mul_op,
    check_rsub_scalar_op,
    check_slice_op,
    check_softmax_op,
    check_trans_op,
    check_unsqueeze_op,
    get_dtype,
)
from .check_npu_ops import check_npu_dtype_cast_op

"""
We try to match torch_cal() pattern, fuse all aten.ops in one triton kernel

inputs:
#  view_7 = torch.ops.aten.view.default(buf34, [11, 12, 256, 256]).npu()
#  buf47 = torch.rand([11, 12, 256, 256], dtype=torch.float16).npu()
#  buf59 = torch.rand([11, 12, 256, 256], dtype=torch.float16).npu()
#  arg107_1 = torch.randint(1, 1000, (11, 256), dtype=torch.int64).npu()
#  buf61 = torch.full([], -65504.0, dtype=torch.float16).npu()

outputs:
convert_element_type_21

def torch_cal(view_7, gather, gather_1, arg107_1, full_default):
    slice_3: "i64[11, 256]" = torch.ops.aten.slice.Tensor(arg107_1, 1, 0, 512)
    unsqueeze_1: "i64[11, 1, 256]" = torch.ops.aten.unsqueeze.default(slice_3, 1)
    unsqueeze_2: "i64[11, 1, 1, 256]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2)
    expand: "i64[11, 1, 256, 256]" = torch.ops.aten.expand.default(unsqueeze_2, [11, 1, 256, 256])
    npu_dtype_cast_1: "f16[11, 1, 256, 256]" = torch.ops.npu.npu_dtype_cast.default(expand, torch.float16)
    sub_1: "f16[11, 1, 256, 256]" = torch.ops.aten.sub.Tensor(1.0, npu_dtype_cast_1)
    npu_dtype_cast_2: "b8[11, 1, 256, 256]" = torch.ops.npu.npu_dtype_cast.default(sub_1, torch.bool)
    where: "f16[11, 1, 256, 256]" = torch.ops.aten.where.self(npu_dtype_cast_2, full_default, sub_1)
    permute_10: "f16[11, 12, 256, 256]" = torch.ops.aten.permute.default(gather_1, [0, 1, 3, 2])
    add_4: "f16[11, 12, 256, 256]" = torch.ops.aten.add.Tensor(gather, permute_10)
    mul_4: "f16[11, 12, 256, 256]" = torch.ops.aten.mul.Tensor(add_4, 0.07216878364870322)
    add_5: "f16[11, 12, 256, 256]" = torch.ops.aten.add.Tensor(view_7, mul_4)
    add_6: "f16[11, 12, 256, 256]" = torch.ops.aten.add.Tensor(add_5, where)
    convert_element_type_20: "f32[11, 12, 256, 256]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)
    # next is softmax
    amax: "f32[11, 12, 256, 1]" = torch.ops.aten.amax.default(convert_element_type_20, [-1], True)
    sub_4: "f32[11, 12, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax)
    exp: "f32[11, 12, 256, 256]" = torch.ops.aten.exp.default(sub_4)
    sum_1: "f32[11, 12, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[11, 12, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1)
    convert_element_type_21: "f16[11, 12, 256, 256]" = torch.ops.prims.convert_element_type.default(div, torch.float16)
    return convert_element_type_21
"""

from .triton_kernel.fused_brc_permute_sum import fused_brc_permute_sum


class BrcPermuteSumOperation(nn.Module):
    def forward(self, view_7, buf47, buf59, arg107_1, buf61):
        return torch.ops.torch_npu_triton.fused_brc_permute_sum(view_7, buf47, buf59, arg107_1, buf61)


class FusedBrcPermuteSum(Pattern):
    _opt_level = OptLevel.level2

    def _match_expand_div_pair(self, div_node: Node) -> Optional[Tuple[Node, Node]]:
        if div_node.op != "call_function" or div_node.target != torch.ops.aten.div.Tensor:
            return None

        _, divisor = div_node.args
        if divisor.op != "call_function" or divisor.target != torch.ops.aten.expand.default:
            return None

        base_input = divisor.args[0]
        expand_shape = divisor.args[1]
        return (divisor, base_input, expand_shape)

    # TODO: currently we do not parse the const value inside the pattern,
    # i.e. 1.0 and 0.07216878364870322 shown in the fused triton kernel.
    def _match_pattern(self, root: Node) -> Optional[List[Node]]:
        # check first slice op
        if not check_slice_op(root):
            return None
        if not (get_dtype(root.args[0]) in (torch.int64, torch.int32)):
            return None
        slice = root
        #
        input0 = None
        input1 = None
        input2 = None
        input3 = slice.args[0]
        input4 = None
        # slice -> unqueeze
        slice_unqueeze = None
        for user in slice.users:
            if not check_unsqueeze_op(user):
                continue
            if not (user.args[1] == 1):
                continue
            slice_unqueeze = user
            break
        if slice_unqueeze == None:
            return None
        # slice -> unqueeze -> unqueeze
        slice_unqueeze2 = None
        for user in slice_unqueeze.users:
            if not check_unsqueeze_op(user):
                continue
            if not (user.args[1] == 2):
                continue
            slice_unqueeze2 = user
            break
        if slice_unqueeze2 == None:
            return None
        # slice -> unqueeze -> unqueeze -> expand
        slice_unqueeze2_expand = None
        for user in slice_unqueeze2.users:
            if not check_expand_op(user):
                continue
            if not ((len(user.args[1]) == 4) and (user.args[1][1] == 1)):
                continue
            slice_unqueeze2_expand = user
            break
        if slice_unqueeze2_expand == None:
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast
        slice_unqueeze2_expand_cast = None
        for user in slice_unqueeze2_expand.users:
            if not check_npu_dtype_cast_op(user):
                continue
            if not (user.args[1] == torch.float16):
                continue
            slice_unqueeze2_expand_cast = user
            break
        if slice_unqueeze2_expand_cast == None:
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub
        slice_unqueeze2_expand_cast_rsub = None
        for user in slice_unqueeze2_expand_cast.users:
            if not check_rsub_scalar_op(user):
                continue
            if not (user.args[1] == 1.0):
                continue
            slice_unqueeze2_expand_cast_rsub = user
            break
        if slice_unqueeze2_expand_cast_rsub == None:
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub -> cast
        slice_unqueeze2_expand_cast_rsub_cast = None
        for user in slice_unqueeze2_expand_cast_rsub.users:
            if not check_npu_dtype_cast_op(user):
                continue
            if not (user.args[1] == torch.bool):
                continue
            slice_unqueeze2_expand_cast_rsub_cast = user
            break
        if slice_unqueeze2_expand_cast_rsub_cast == None:
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub -> cast -> masked_fill
        slice_unqueeze2_expand_cast_rsub_mskfil = None
        for user in slice_unqueeze2_expand_cast_rsub.users:
            if not check_mask_fill_op(user):
                continue
            if not (
                (user.args[0] == slice_unqueeze2_expand_cast_rsub)
                and (user.args[1] == slice_unqueeze2_expand_cast_rsub_cast)
            ):
                continue
            slice_unqueeze2_expand_cast_rsub_mskfil = user
            break
        if slice_unqueeze2_expand_cast_rsub_mskfil == None:
            return None
        input4 = slice_unqueeze2_expand_cast_rsub_mskfil.args[2]
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub -> cast -> masked_fill
        # -> add
        add = None
        add_other = None
        for user in slice_unqueeze2_expand_cast_rsub_mskfil.users:
            if not check_add_op(user):
                continue
            add = user
            add_other = next((e for e in user.args if e != slice_unqueeze2_expand_cast_rsub_mskfil), None)
            break
        if add == None:
            return None
        if add_other == None:
            return None
        # -> add <- add
        if not check_add_op(add_other):
            return None
        add_add = add_other
        # -> add <- add <- mul
        add_add_mul = None
        for arg in add_add.args:
            if not check_mul_op(arg):
                input0 = arg
                continue
            # FIXME: Do we need to check the 2nd argument which is a scalar?
            add_add_mul = arg
            break
        # -> add <- add <- mul <- add
        add_add_mul_add = None
        for arg in add_add_mul.args:
            if not check_add_op(arg):
                continue
            add_add_mul_add = arg
            break
        if add_add_mul_add == None:
            return None
        # -> add <- add <- mul <- add <- transpose
        add_add_mul_add_trans = None
        for arg in add_add_mul_add.args:
            if not check_trans_op(arg):
                input1 = arg
                continue
            if not ((arg.args[1] == -1) and (arg.args[2] == -2)):
                continue
            add_add_mul_add_trans = arg
            break
        input2 = add_add_mul_add_trans.args[0]
        # -> add -> softmax
        add_softmax = None
        for user in add.users:
            if not check_softmax_op(user):
                continue
            if not (user.args[1] == -1):
                continue
            add_softmax = user
            break
        if add_softmax == None:
            return None
        final = add_softmax
        return [input0, input1, input2, input3, input4, final]

    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_brc_permute_sum", BrcPermuteSumOperation())

        for node in list(graph.nodes):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue

            (input0, input1, input2, input3, input4, final) = matched_nodes
            with graph.inserting_before(final):
                fused_node = graph.call_module(
                    "npu_triton_fused_brc_permute_sum",
                    args=(input0, input1, input2, input3, input4),
                )
            final.replace_all_uses_with(fused_node)
            changed = True

            nodes_to_remove = [input0, input1, input2, input3, input4, final]
            for node in nodes_to_remove:
                if isinstance(node, ()) and len(node.users) == 0:
                    graph.erase_node(node)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed
