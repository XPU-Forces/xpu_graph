import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_cat_op, check_meta_2d, check_mul_op, check_sum_op


def find_dotprod_pattern(gm):
    # Dictionary to store mapping from source nodes to sum nodes
    dotprod_dict = {}
    for node in gm.graph.nodes:
        if not check_sum_op(node):
            continue
        ### Identify sum operation: input 3D, output 2D ###
        if not check_meta_2d(node):
            continue
        # check sum dim
        if node.args[1] != [1]:
            continue
        # don't keep dim
        if len(node.args) > 2 and node.args[2] == True:
            continue
        if len(node.users) != 1:
            continue

        mul_node = node.args[0]
        if not check_mul_op(mul_node):
            continue
        if len(mul_node.users) != 1:
            continue
        if not isinstance(mul_node.args[0], fx.Node) or not isinstance(mul_node.args[1], fx.Node):
            continue
        x = mul_node.args[0]
        y = mul_node.args[1]
        if x.meta["val"].shape[1:] != y.meta["val"].shape[1:]:
            continue
        if node not in dotprod_dict:
            dotprod_dict[node] = (mul_node.args[0], mul_node.args[1])
    return dotprod_dict


def check_enable(sum_inputs, sum_dict):
    for sum_node in sum_inputs:
        if sum_node not in sum_dict:
            return False
    return True


def get_mul_inputs(sum_inputs):
    mul_inputs = []
    for sum_node in sum_inputs:
        mul_node = sum_node.args[0]
        mul_inputs.append((mul_node.args[0], mul_node.args[1]))
    return mul_inputs


class FusedDotCat(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        if not hasattr(graph_module, "mul_sum_cat_replacement"):
            graph_module.add_submodule("mul_sum_cat_replacement", self.target_mod())

        dotprod_dict = find_dotprod_pattern(graph_module)
        for node in graph_module.graph.nodes:
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat or cat_axis != 1:
                continue
            if len(node.args[0]) < 2:
                continue
            if any(x not in dotprod_dict for x in node.args[0]):
                continue
            mul_inputs = [dotprod_dict[dot_node] for dot_node in node.args[0]]
            x_list = [x for x, _ in mul_inputs]
            y_list = [y for _, y in mul_inputs]

            bsz, s1, s2 = x_list[0].meta["val"].shape
            bsz = max(bsz, y_list[0].meta["val"].shape[0])
            if any(x.meta["val"].shape[1:] != (s1, s2) for x in x_list[1:]):
                continue
            if any(max(x.meta["val"].shape[0], y.meta["val"].shape[0]) != bsz for x, y in mul_inputs):
                continue

            with graph_module.graph.inserting_before(node):
                new_node = graph_module.graph.call_module(
                    "mul_sum_cat_replacement",
                    args=(x_list, y_list),
                )
            node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(node)
            changed = True

        return changed
