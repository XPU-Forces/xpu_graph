import torch
import torch.fx as fx
from torchair.scope._scope import *

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

# WARNING(liuyuan): The `.default` overload is very important to help use make the node has side effect so it won't be eleminated. See #345
__SCOPE_ENTER__ = torch.ops.air.scope_enter.default
__SCOPE_EXIT__ = torch.ops.air.scope_exit.default
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
            graph.call_function(__SCOPE_ENTER__, args=__ARGS__)
        with graph.inserting_before(next(iter(reversed(graph.nodes)))):
            graph.call_function(__SCOPE_EXIT__)

        return True
