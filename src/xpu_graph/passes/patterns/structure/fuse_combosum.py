import operator
from typing import Callable

import torch
from torch import fx, nn

from xpu_graph import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup

from ..utils.check_ops import (
    check_cat_op,
    check_getitem_op,
    check_meta_2d,
    check_slice_op,
    check_stack_op,
    check_sum_op,
    get_actual_node,
)
from ..utils.match_sub_list import match_sub_list


def find_slice_cat(node):
    for name in ["fuse_slice_cat", "fuse_slice_cat_v2"]:
        if name in node.name:
            return True
    return False


def find_sum3dinp_input(candidates):
    sum_input_dict = {}
    for sum_node in candidates:
        if not check_meta_2d(sum_node):
            continue
        # don't keep dim
        if len(sum_node.args) > 2:
            continue
        input_node = get_actual_node(sum_node, 0)
        if not check_getitem_op(input_node):
            continue
        src_node = input_node.args[0]
        if not find_slice_cat(src_node):
            continue
        dim = sum_node.args[1]
        # only support dim=1/2
        if dim not in [[1], [2]]:
            continue
        src_node = input_node.args[0]
        key = src_node.name + str(dim)
        if key not in sum_input_dict:
            sum_input_dict[key] = []
        sum_input_dict[key].append(sum_node)
    return sum_input_dict


def partly_topo_sort(gm: fx.Graph, node: fx.Node):
    import queue

    que = queue.Queue()
    que.put(node)
    while not que.empty():
        cur = que.get()
        for user in cur.users:
            if user < cur:
                cur.append(user)
                que.put(user)


class ComboSum3dInp(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1
    """
    #input: 3d output:2d
    sum1(a), sum2(b)-> combo_sum(a,b) #after emb_cat
    """

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        changed = False

        graph_module.add_submodule(
            "combo_sum",
            self.target_mod(),
        )
        candidates = [
            node
            for node in graph_module.graph.nodes
            if (node.op == "call_function" or node.op == "call_module")
            and node.target == torch.ops.aten.sum.dim_IntList
        ]
        sum_input_dict = find_sum3dinp_input(candidates)
        for key, sum_nodes in sum_input_dict.items():
            sum_node = sum_nodes[-1]
            with graph_module.graph.inserting_after(sum_node):
                new_nodes = graph_module.graph.call_module(
                    "combo_sum",
                    args=([s.args[0] for s in sum_nodes], sum_node.args[1]),
                )

            for idx, ori_node in enumerate(sum_nodes):
                with graph_module.graph.inserting_after(new_nodes):
                    idx_node = graph_module.graph.call_function(operator.getitem, args=(new_nodes, idx))
                ori_node.replace_all_uses_with(idx_node)
                partly_topo_sort(graph_module, idx_node)
                graph_module.graph.erase_node(ori_node)
            changed = True

        return changed
