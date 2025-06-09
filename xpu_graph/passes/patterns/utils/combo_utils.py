from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn


def find_last_node_in_list(gm: fx.GraphModule, node_list: list[fx.Node]) -> fx.Node:
    """
    Given a list of nodes, find the one that appears last in the graph's topological order.
    """
    node_set = set(node_list)  # Faster lookup
    last_node = None

    for node in gm.graph.nodes:
        if node in node_set:
            last_node = node  # Update when we see a matching node

    return last_node


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


def extract_nodes_from_args_kwargs(args, kwargs):
    """
    从给定的 args 和 kwargs 中递归提取所有 fx.Node 实例。
    """
    nodes = []

    def recurse(item):
        if isinstance(item, fx.Node):
            nodes.append(item)
        elif isinstance(item, (list, tuple)):
            for elem in item:
                recurse(elem)
        elif isinstance(item, dict):
            for value in item.values():
                recurse(value)
        # 其他类型（如 int、float、str 等）不处理

    recurse(args)
    recurse(kwargs)
    return nodes


def get_ancestors(node):
    """
    找给定node的所有祖先
    """
    stack = [node]
    ancestors = []
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        if node is None:
            continue
        if node.op == "placeholder":
            continue
        ancestors.append(node)
        stack += extract_nodes_from_args_kwargs(node.args, node.kwargs)
    if len(ancestors) > 0:
        # remove node
        ancestors = ancestors[1:]
    return ancestors


def find_dep(nodes, dep_func):
    """
    给定依赖函数和节点队列, 返回分组好的节点.
    """
    groups = []

    for node in nodes:
        placed = False
        for group in groups:
            if any(dep_func(node, other) for other in group):
                continue
            group.append(node)
            placed = True
            break
        if not placed:
            groups.append([node])

    return groups
