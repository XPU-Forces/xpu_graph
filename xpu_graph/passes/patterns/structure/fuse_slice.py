from typing import Callable

import torch
from torch import fx, nn

# import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup

from ..utils.check_ops import check_slice_op, check_stack_op, get_shape
from ..utils.submodule_manager import register_new_submodule


def custom_getitem(tensor_list, index):
    return tensor_list[index]


def divide_nodes_in_slice_len(nodes):
    """
    Groups nodes based on the length of their slice operation.

    Args:
        nodes (list): A list of nodes, where each node represents a slice operation.

    Returns:
        dict: A dictionary where keys are slice lengths (slice_end - slice_start),
              and values are lists of tuples, each containing a node and its slice start index.
    """
    divide_nodes = {}
    for n in nodes:
        slice_len = n.args[3] - n.args[2]
        if slice_len not in divide_nodes:
            divide_nodes[slice_len] = []
        divide_nodes[slice_len].append((n, n.args[2]))
    return divide_nodes


def find_slice_nodes(graph_module):
    """
    Identifies and groups slice operation nodes in a given graph module.

    Args:
        graph_module: A graph module containing computation nodes.

    Returns:
        dict: A dictionary where keys are parent nodes, and values are lists of
              associated slice operation nodes.
    """
    candi_nodes = {}
    for node in graph_module.graph.nodes:
        if not check_slice_op(node):
            continue
        if not isinstance(node.args[1], int):
            continue
        if not isinstance(node.args[2], int):
            continue
        if not isinstance(node.args[3], int):
            continue

        # slice dim must be lowest dim
        src_node = node.args[0]
        axis = node.args[1]
        if axis != -1:
            dim = len(get_shape(src_node))
            if axis != dim - 1:
                continue

        # Skip slice nodes that are directly connected to the output node.
        if len(node.users) == 1:
            if next(iter(node.users)).target == "output":
                continue
        if node.args[0] not in candi_nodes:
            candi_nodes[node.args[0]] = []
        candi_nodes[node.args[0]].append(node)
    return candi_nodes


# one src node slice to multi dst nodes
class FusedSlice(Pattern):
    _pattern_group = PatternGroup.GROUP1

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        candi_nodes = find_slice_nodes(graph_module)

        for src_node, nodes in candi_nodes.items():
            # grouped by output_len
            divide_nodes = divide_nodes_in_slice_len(nodes)

            for slice_len, nodes2 in divide_nodes.items():
                if len(nodes2) < 3:
                    continue
                start_indices = [n[1] for n in nodes2]
                replace_n = [n[0] for n in nodes2]
                # output: [num, src_node[0], slice_len]
                with graph_module.graph.inserting_before(replace_n[0]):
                    module_name = register_new_submodule(
                        graph_module,
                        "mlu_triton_slice",
                        self.target_mod,
                        args=(start_indices,),
                    )
                    new_node = graph_module.graph.call_module(
                        module_name,
                        args=(src_node, slice_len),
                    )
                # TODO: put in to fused_slice_module
                for idx, n in enumerate(replace_n):
                    # output: [src_node[0], slice_len]
                    with graph_module.graph.inserting_before(n):
                        new_n = graph_module.graph.create_node(
                            op="call_function",
                            target=custom_getitem,
                            args=(new_node, idx),
                            name=f"getitem_node_{new_node.name}_{n.name}",
                        )
                    n.replace_all_uses_with(new_n)
                    graph_module.graph.erase_node(n)
                changed = True

        return changed
