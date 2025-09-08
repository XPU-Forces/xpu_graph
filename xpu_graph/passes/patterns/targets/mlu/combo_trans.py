import operator
from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.fx_utils import FxStage
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

from ...utils.check_ops import get_shape 

COMBINE_LEN = 4

def check_trans_list(nodes, expected_target, dim):
    """
    检查所有节点的target是否相同，并且形状都是2维且一样
    """
    for node in nodes:
        if node.target != expected_target:
            return False

        if len(node.users) > 1:
            return False

    # check shapes
    shapes = [get_shape(node) for node in nodes]
    if not all(len(shape) == dim for shape in shapes):
        return False
    first_shape = shapes[0]
    return all(shape == first_shape for shape in shapes)


def replace_trans(graph_module, node, input_idx):
    graph = graph_module.graph
    original_tensors = [n.args[0] for n in node.args[input_idx]]
    num_tensors = len(original_tensors)
    with graph.inserting_before(node):
        stack_node = graph.call_function(
            torch.ops.aten.stack.default,
            args=(original_tensors,),
            kwargs={"dim": 0}
        )
        
        transpose_node = graph.call_function(
            torch.ops.aten.transpose.int,
            args=(stack_node, -1, -2)
        )
        unbind_node = graph.call_function(
            torch.ops.aten.unbind.int, args=(transpose_node,)
        )
        for i in range(num_tensors):
            n = node.args[input_idx][i]
            new_n = graph.call_function(
                operator.getitem, args=(unbind_node, i)
            )
            n.replace_all_uses_with(new_n)


def replace_trans1(graph_module, node, input_idx, ori_shape):
    graph = graph_module.graph
    original_tensors = [n.args[0] for n in node.args[input_idx]]
    num_tensors = len(original_tensors)
    with graph.inserting_before(node):
        cat_node = graph.call_function(
            torch.ops.aten.cat.default,
            args=(original_tensors,),
            kwargs={"dim": 0}
        )
        transposed_cat_node = graph.call_function(
            torch.ops.aten.t.default,
            args=(cat_node,)
        )
        view_node = graph.call_function(
            torch.ops.aten.view.default,
            args=(cat_node, [num_tensors, ori_shape[0], -1])
        )
        unbind_node = graph.call_function(torch.unbind, args=(view_node,), kwargs={"dim": 0})
        for i in range(num_tensors):
            n = node.args[input_idx][i]
            new_n = graph.call_function(
                operator.getitem, args=(unbind_node, i)
            )
            n.replace_all_uses_with(new_n)
            graph_module.graph.erase_node(n)


class FusedComboTrans(Pattern):
    _pattern_group = PatternGroup.GROUP1
    _support_stages = [
        FxStage.inference,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        target_module = [
            "fused_combo_bmm",
        ]
        candidates = [
            node
            for node in graph_module.graph.nodes
            if (node.op == "call_function" or node.op == "call_module")
            and node.target in target_module 
        ]
        for node in candidates:
            if node.target == "fused_combo_bmm":
                for i in range(3):
                    if len(node.args[i]) < COMBINE_LEN:
                        continue
                    nodes = node.args[i]
                    first_node = nodes[0] 
                    if first_node == None:
                        continue
                    if all(first_node.name == n.name for n in nodes):
                        continue
                    if check_trans_list(nodes, torch.ops.aten.t.default, 2) or check_trans_list(nodes, torch.ops.aten.transpose.int, 3):
                        replace_trans(graph_module, node, i)
                        changed = True

        return changed
