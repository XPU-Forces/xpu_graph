import operator
from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

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
    def forward(self, input_list, weight_list, bias_list, act: str):
        output = None
        if len(input_list[0].shape) == 3:
            output = self.forward_bmm(input_list, weight_list, bias_list, act)
            if bias_list[0] is None:
                bias_batch = None
            else:
                bias_list = [bias.unsqueeze(0) if len(bias.shape) == 1 else bias for bias in bias_list]
                bias_batch = torch.stack(bias_list, dim=0)  # Stack all biases along a new dimension
            # Add bias batch if available
            if bias_batch is not None:
                output = output + bias_batch
        else:
            if bias_list[0] is None:
                output = self.forward_mm(input_list, weight_list, [], [], [], act)
            elif len(bias_list[0].shape) == 1:
                output = self.forward_mm(input_list, weight_list, [], bias_list, [], act)
            else:
                beta = [1.0] * len(input_list)
                output = self.forward_mm(input_list, weight_list, bias_list, [], beta, act)

        # Apply activation function if specified
        if act == "relu":
            output = torch.relu(output)
        elif act == "gelu":
            output = torch.gelu(output)
        elif act == "silu":
            output = torch.silu(output)
        elif act == "sigmoid":
            output = torch.sigmoid(output)

        return output

    def forward_bmm(self, input_list, weight_list, bias_list, act: str):
        input_batch = torch.stack(input_list, dim=0)
        weight_batch = torch.stack(weight_list, dim=0)
        T, B, K, N = weight_batch.shape
        M = input_list[0].shape[1]
        output = torch.bmm(input_batch.view(-1, M, K), weight_batch.view(-1, K, N)).view(T, B, M, N)
        return output

    def forward_mm(self, input_list, weight_list, c_list, bias_list, beta, act: str):
        output_list = torch.ops.torch_mlu.grouped_gemm(
            input_list,
            weight_list,
            c=c_list,
            bias=bias_list,
            alpha=[],
            beta=beta,
            trans_a=False,
            trans_b=False,
        )
        output = torch.stack(output_list, dim=0)
        return output

        """
        # Fallback case: ordinary matmul + bias for each pair
        else:
            if len(input_list) == 1:
                input_list = input_list * len(weight_list)
            output = [
                torch.matmul(input, weight) + bias
                for input, weight, bias in zip(input_list, weight_list, bias_list)
            ]
        """


def replace_node(graph_module, nodes):
    new_input = [n.input1 for n in nodes]
    new_weight = [n.input2 for n in nodes]
    new_bias = [n.bias for n in nodes]
    act = nodes[0].act

    if len(new_weight) < COMBINE_LEN:
        return
    with graph_module.graph.inserting_after(find_last_node_in_list(graph_module, new_input + new_weight + new_bias)):
        new_node = graph_module.graph.call_module(
            "fused_combine_bmm",
            args=(new_input, new_weight, new_bias, act),
        )
    with graph_module.graph.inserting_after(new_node):
        for idx, n in enumerate(nodes):
            new_n = graph_module.graph.call_function(operator.getitem, args=(new_node, idx))
            n.node.replace_all_uses_with(new_n)
            partly_topo_sort(graph_module, new_n)
    graph_module.graph.lint()
    graph_module.recompile()


def combine_matmul(graph_module, candidates):
    changed = False
    group_by_shape = {}
    # split mm by input&weight&bias's shape and activation mode
    for n in candidates:
        mm_desc = get_node_desc(n)
        if mm_desc == None:
            continue
        key = (
            mm_desc.input1_shape,
            mm_desc.input2_shape,
            mm_desc.bias_shape,
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

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        graph_module.add_submodule("fused_combine_bmm", FusedCombineBmm())
        # split mm by difference module
        target_module = [
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
            "mlu_tmo_fused_matmul_replacement",
            "mlu_tmo_fused_matmul_add_replacement",
            "mlu_tmo_fused_matmul_act_replacement",
            "mlu_tmo_fused_matmul_add_act_replacement",
            "mlu_tmo_fused_bmm_replacement",
            "mlu_tmo_fused_bmm_add_replacement",
            "mlu_tmo_fused_bmm_act_replacement",
            "mlu_tmo_fused_bmm_add_act_replacement",
        ]
        for module in target_module:
            candidates = [
                node
                for node in graph_module.graph.nodes
                if (node.op == "call_function" or node.op == "call_module") and node.target == module
            ]
            if len(candidates) < COMBINE_LEN:
                continue
            changed = changed | combine_matmul(graph_module, candidates)

        return changed
