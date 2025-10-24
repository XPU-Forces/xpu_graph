import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_op, get_shape
from ..utils.shape_utils import same_shape

FUSED_ADDN_LIMIT = 4


class FusedAddN(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference, FxStage.forward, FxStage.pregrad, FxStage.backward]
    """
    a = add(x1, x2)
    b = add(a, x3)
    c = add(b, x4)
    -->
    stack([x1, x2, x3, x4]).sum(dim=[0])

    # Note: only left adder is considered because of accumulate-order
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        candidates = [
            node
            for node in reversed(graph_module.graph.nodes)
            if node.op == "call_function"
            and check_op(node, torch.ops.aten.add.Tensor)
            and isinstance(node.args[1], fx.Node)
            and isinstance(node.args[1].meta["val"], torch.Tensor)
            and same_shape(get_shape(node.args[0]), get_shape(node.args[1]))
            and node.args[0].meta["val"].dtype == node.args[1].meta["val"].dtype
        ]

        for add in candidates:
            n = add
            add_list = [n.args[1]]
            delete_list = [n]
            while True:
                inp0 = n.args[0]
                if inp0 in candidates and len(inp0.users) == 1:
                    add_list.append(inp0.args[1])
                    delete_list.append(inp0)
                    n = inp0
                else:
                    add_list.append(inp0)
                    break
            if len(add_list) < FUSED_ADDN_LIMIT:
                continue

            add_list = list(reversed(add_list))

            with graph_module.graph.inserting_before(add):
                stack_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.stack.default,
                    args=(add_list,),
                    name=add.name + "_fusedaddn_stack",
                )
                sum_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.sum.dim_IntList,
                    args=(stack_node, [0]),
                    name=add.name + "_fusedaddn_sum",
                )
                add.replace_all_uses_with(sum_node)
                changed = True
                for add_node in delete_list:
                    if len(add_node.users) == 0:
                        graph_module.graph.erase_node(add_node)
        return changed
