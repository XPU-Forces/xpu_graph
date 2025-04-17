import torch
from torch import nn, fx
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from typing import Callable
from ..utils.submodule_manager import register_new_submodule
import operator


def fuse_multiple_cat1(graph_module: fx.GraphModule, target_mod):
    changed = False
    cat_pattern_all = {}
    for node in graph_module.graph.nodes:
        if node.target == "fuse_slice_cat":
            if node.args[0] not in cat_pattern_all:
                cat_pattern_all[node.args[0]] = [(node, node.args[1])]
            else:
                cat_pattern_all[node.args[0]].append((node, node.args[1]))
    for src_node, nodes_slice_params in cat_pattern_all.items():
        if len(nodes_slice_params) < 2:
            continue
        ori_nodes = []
        slice_params = []
        for v in nodes_slice_params:
            ori_node, slice_param = v
            ori_nodes.append(ori_node)
            slice_params.append(slice_param)
        with graph_module.graph.inserting_before(ori_nodes[0]):

            module_name = register_new_submodule(
                graph_module,
                "fuse_slice_cat_v2",
                target_mod,
                args=(slice_params,),
            )
            new_nodes = graph_module.graph.call_module(
                module_name,
                args=(src_node, slice_params),
            )
        for idx, ori_node in enumerate(ori_nodes):
            with graph_module.graph.inserting_before(ori_node):
                idx_node = graph_module.graph.call_function(
                    operator.getitem, args=(new_nodes, idx)
                )
            ori_node.replace_all_uses_with(idx_node)
            graph_module.graph.erase_node(ori_node)
        changed = True
    return changed


def fuse_multiple_cat2(graph_module: fx.GraphModule, target_mod):
    changed = False
    cat_pattern_all = {}
    for node in graph_module.graph.nodes:
        if "fuse_slice_cat_v2" in node.name:
            if node.args[0] not in cat_pattern_all:
                cat_pattern_all[node.args[0]] = [(node, node.args[1])]
            else:
                cat_pattern_all[node.args[0]].append((node, node.args[1]))
    for src_node, nodes_slice_params in cat_pattern_all.items():
        if len(nodes_slice_params) < 2:
            continue
        ori_nodes = []
        slice_params = []
        getitem_nodes = []
        for v in nodes_slice_params:
            ori_node, slice_param = v
            ori_nodes.append(ori_node)
            slice_params += slice_param
            getitem_nodes_for_this_node = [
                user
                for user in ori_node.users
                if user.op == "call_function" and user.target == operator.getitem
            ]
            getitem_nodes_for_this_node.sort(key=lambda node: node.args[1])
            getitem_nodes += getitem_nodes_for_this_node
        with graph_module.graph.inserting_before(ori_nodes[0]):
            module_name = register_new_submodule(
                graph_module,
                "fuse_slice_cat_v2",
                target_mod,
                args=(slice_params,),
            )
            new_nodes = graph_module.graph.call_module(
                module_name,
                args=(src_node, slice_params),
            )
        for idx, getitem_node in enumerate(getitem_nodes):
            with graph_module.graph.inserting_after(new_nodes):
                idx_node = graph_module.graph.call_function(
                    operator.getitem, args=(new_nodes, idx)
                )
            getitem_node.replace_all_uses_with(idx_node)
        changed = True

    return changed


class FusedMultipleSliceCat(Pattern):

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        changed = False
        # merge multiple "fuse_slice_cat" with the same src_node to one "fuse_slice_cat"
        # fused_slice_cat(x, [A, B]) + fused_slice_cat(x, [C, D]) ->fused_slice_cat_v2(x, [[A, B], [C, D]])
        changed = changed | fuse_multiple_cat1(graph_module, self.target_mod)
        # fused_slice_cat_v2(x, [[A, B], [C, D]]) + fused_slice_cat_v2(x, [[E, F], [D, H]]) ->fused_slice_cat_v2(x, [[A, B], [C, D], [E, F], [D, H]])
        changed = changed | fuse_multiple_cat2(graph_module, self.target_mod)
        return changed
