import torch
import torch.fx as fx
from torchair.scope._scope import *

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

__SCOPE_ENTER__ = torch.ops.air.scope_enter
__SCOPE_EXIT__ = torch.ops.air.scope_exit
__ARGS__ = ["_super_kernel_scope", "_super_kernel_options"], ["super_kernel_scoped_by_xpu_graph", ""]


def is_scope_enter(node: fx.Node):
    return node.op == "call_function" and node.target == __SCOPE_ENTER__


def is_scope_exit(node: fx.Node):
    return node.op == "call_function" and node.target == __SCOPE_EXIT__


class ScopedSuperKernel(Pattern):
    _opt_level = OptLevel.level1

    def process(self, gm: fx.GraphModule):
        graph = gm.graph

        # NOTE(liuyuan): no need to add nodes for super kernel scope.
        if graph.find_nodes(op="call_function", target=__SCOPE_ENTER__):
            return False

        # NOTE(liuyuan): We just scope the entire graph for now.
        with graph.inserting_before(next(iter(graph.nodes))):
            scope_node = graph.call_function(__SCOPE_ENTER__, args=__ARGS__)
            # NOTE(liuyuan): to bypass the graph.eliminate_dead_code
            scope_node.is_impure = lambda: True
        with graph.inserting_before(next(iter(reversed(graph.nodes)))):
            scope_node = graph.call_function(__SCOPE_EXIT__)
            # NOTE(liuyuan): to bypass the graph.eliminate_dead_code
            scope_node.is_impure = lambda: True

        return True
