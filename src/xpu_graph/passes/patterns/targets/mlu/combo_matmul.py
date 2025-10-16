import operator
from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.fx_utils import FxStage
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

from ...utils.check_ops import (  # get_shape,
    check_act_op,
    check_add_op,
    check_addmm_op,
    check_bmm_op,
    check_mm_op,
    check_t_op,
    check_trans_op,
    check_view,
)
from ...utils.combo_utils import *
from .combo_matmul_utils import *

from xpu_graph.passes.patterns.utils.shape_utils import SymShapeManager
import torch.nn.functional as F


TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node
COMBINE_LEN = 4

from typing import Optional, Union


def all_same_tensor(tensor_list):
    if not tensor_list:
        return True
    first = tensor_list[0]
    return all(t is first for t in tensor_list)


class FusedCombineBmm(nn.Module):
    def forward(
        self, input_list, weight_list, bias_list, act: str, trans_a: bool, trans_b: bool
    ):
        use_groupgemm = True
        output = None

        if len(input_list[0].shape) == 2 and use_groupgemm:
            output = self.forward_groupgemm(
                input_list, weight_list, bias_list, trans_a, trans_b
            )
        else:
            output = self.forward_bmm(
                input_list, weight_list, bias_list, trans_a, trans_b
            )

        # Apply activation function if specified

        activations = {
            "relu": torch.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "sigmoid": torch.sigmoid,
        }
        if act in activations:
            act_func = activations.get(act.lower())
            output = act_func(output)

        return output

    def forward_bmm(self, input_list, weight_list, bias_list, trans_a, trans_b):
        input_batch = torch.stack(input_list, dim=0)
        if trans_a:
            input_batch = input_batch.transpose(-1, -2)

        weight_batch = torch.stack(weight_list, dim=0)
        if trans_b:
            weight_batch = weight_batch.transpose(-1, -2)

        input_shape = input_batch.shape
        weight_shape = weight_batch.shape
        M = input_shape[-2]
        K = weight_shape[-2]
        N = weight_shape[-1]

        if len(weight_shape) == 4:
            input_batch = input_batch.view(-1, M, K)
            weight_batch = weight_batch.view(-1, K, N)

        if bias_list[0] is not None:
            bias_batch = torch.stack(bias_list)
            # bias: [N]
            if len(bias_list[1].shape) == 1:
                # bias_batch: [T, 1, N]
                bias_batch = bias_batch.unsqueeze(1)
            if len(bias_batch.shape) == 4:
                bias_batch = bias_batch.view(-1, M, N)
            output = torch.bmm(input_batch, weight_batch) + bias_batch
        else:
            output = torch.bmm(input_batch, weight_batch)


        if len(weight_shape) == 4:
            output = output.view(weight_shape[0], input_shape[1], M, N)
        return output

    def forward_groupgemm(
        self, input_list, weight_list, bias_list, trans_a: bool, trans_b: bool
    ):
        processed_inputs = [
            i.contiguous() if not i.is_contiguous() else i for i in input_list
        ]
        processed_weights = [
            w.contiguous() if not w.is_contiguous() else w for w in weight_list
        ]
        args = [processed_inputs, processed_weights]
        kwargs = {"trans_a": trans_a, "trans_b": trans_b}
        if bias_list[0] is not None:
            if len(bias_list[0].shape) == 1:
                kwargs["bias"] = bias_list
            else:
                beta = [1.0] * len(input_list)
                kwargs["beta"] = beta
                kwargs["c"] = bias_list
        #output_list = torch.ops.torch_mlu.grouped_gemm(*args, **kwargs)
        #output = torch.stack(output_list, dim=0)
        output = torch.ops.torch_mlu.grouped_gemm(*args, **kwargs)
        output = output.view(len(input_list), -1, output.shape[-1])
        return output


def replace_node(graph_module, nodes):
    new_input = [n.input1 for n in nodes]
    new_weight = [n.input2 for n in nodes]
    new_bias = [n.bias for n in nodes]
    trans_a = nodes[0].input1_trans
    trans_b = nodes[0].input2_trans
    act = nodes[0].act

    if len(new_weight) < COMBINE_LEN:
        return
    with graph_module.graph.inserting_after(
        find_last_node_in_list(graph_module, new_input + new_weight + new_bias)
    ):
        new_node = graph_module.graph.call_module(
            "fused_combo_bmm",
            args=(new_input, new_weight, new_bias, act, trans_a, trans_b),
        )
    with graph_module.graph.inserting_after(new_node):
        unbind_node = graph_module.graph.call_function(
            torch.ops.aten.unbind.int, args=(new_node,)
        )
    with graph_module.graph.inserting_after(unbind_node):
        for idx, n in enumerate(nodes):
            new_n = graph_module.graph.call_function(
                operator.getitem, args=(unbind_node, idx)
            )
            if n.extra_match == True:
                next_node = next(iter(n.node.users))
                next_node.replace_all_uses_with(new_n)
            else:
                n.node.replace_all_uses_with(new_n)
            partly_topo_sort(graph_module, new_n)
    graph_module.graph.lint()
    graph_module.recompile()


def combo_matmul(graph_module, candidates, sym_shape_manager):
    changed = False
    group_by_shape = {}
    # split mm by input&weight&bias's shape and activation mode
    for n in candidates:
        mm_desc = get_node_desc(n)
        if mm_desc == None:
            continue

        key = (
            mm_desc.input1_trans,
            mm_desc.input2_trans,
            tuple(sym_shape_manager.rebind_shape(mm_desc.input1_shape)),
            tuple(sym_shape_manager.rebind_shape(mm_desc.input2_shape)),
            (
                (
                    None
                    if mm_desc.bias == None
                    else tuple(sym_shape_manager.rebind_shape(mm_desc.bias_shape))
                ),
            ),
            mm_desc.act,
        )
        if key not in group_by_shape:
            group_by_shape[key] = []
        group_by_shape[key].append(mm_desc)

    for key1, group_nodes in group_by_shape.items():
        group_by_input = find_dep(group_nodes, has_mm_dependency)
        for nodes in group_by_input:
            if len(nodes) < COMBINE_LEN:
                continue
            changed = True
            replace_node(graph_module, nodes)
    return changed


class FusedCombineMatMul(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1
    _support_stages = [
        FxStage.inference,
        #FxStage.forward,
        #FxStage.backward,
    ]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        graph_module.add_submodule("fused_combo_bmm", FusedCombineBmm())
        sym_shape_manager = SymShapeManager(graph_module.graph)
        # split mm by difference module
        target_module = [
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
            "custom_batch_dense_layer_replacement",
            "custom_dense_layer_replacement",
        ]
        for module in target_module:
            candidates = [
                node
                for node in graph_module.graph.nodes
                if (node.op == "call_function" or node.op == "call_module")
                and node.target == module
            ]
            if len(candidates) < COMBINE_LEN:
                continue
            changed = changed | combo_matmul(
                graph_module, candidates, sym_shape_manager
            )

        return changed
