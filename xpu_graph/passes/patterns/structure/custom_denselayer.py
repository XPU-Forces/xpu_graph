from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import (
    check_act_op,
    check_addmm_op,
    check_baddbmm_op,
    check_bmm_op,
    check_mm_op,
    check_t_op,
    check_trans_op,
    is_exclusively_used,
)


def check_weight_trans(node):
    weight_trans = False
    if check_trans_op(node):
        trans_param = (node.args[1], node.args[2])
        if trans_param in [(0, 1), (1, 0), (-2, -1), (-1, -2)]:
            weight_trans = True
            node = node.args[0]
    elif check_t_op(node):
        weight_trans = True
        node = node.args[0]
    return weight_trans, node


def match_mm_act(node):
    is_act, act_str = check_act_op(node)
    if not is_act:
        act_str = "none"
        linear_node = node
    else:
        linear_node = node.args[0]
        if not is_exclusively_used(linear_node, node):
            return False, []
    is_mm, inputs, weights = check_mm_op(linear_node)
    if is_mm:
        bias = None
    else:
        is_addmm, bias, inputs, weights = check_addmm_op(linear_node)
        if not is_addmm:
            return False, []
    weight_trans, weights = check_weight_trans(weights)
    return True, [inputs, weights, weight_trans, bias, act_str]


def match_bmm_act(node):
    is_act, act_str = check_act_op(node)
    if not is_act:
        act_str = "none"
        linear_node = node
    else:
        linear_node = node.args[0]
        if not is_exclusively_used(linear_node, node):
            return False, []
    is_mm, inputs, weights = check_bmm_op(linear_node)
    if is_mm:
        bias = None
    else:
        is_addmm, bias, inputs, weights = check_baddbmm_op(linear_node)
        if not is_addmm:
            return False, []
    weight_trans, weights = check_weight_trans(weights)
    return True, [inputs, weights, weight_trans, bias, act_str]


class CustomDenseLayer(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, constraint_fn, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod
        self.constraint_fn = constraint_fn

    def process(self, graph_module: fx.GraphModule) -> bool:
        return False
        fast_act = True if self._opt_level == OptLevel.level3 else False
        if not hasattr(graph_module, "custom_dense_layer_replacement"):
            graph_module.add_submodule("custom_dense_layer_replacement", self.target_mod(fast_act))
        changed = False
        for node in reversed(graph_module.graph.nodes):
            is_match, mm_params = match_mm_act(node)
            if is_match and self.constraint_fn(*fx.map_arg(mm_params, lambda x: x.meta["val"])):
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module("custom_dense_layer_replacement", args=tuple(mm_params))
                node.replace_all_uses_with(new_node)
                act_str = mm_params[-1]
                if act_str != "none":
                    before_act = node.args[0]
                    graph_module.graph.erase_node(node)
                    graph_module.graph.erase_node(before_act)
                else:
                    graph_module.graph.erase_node(node)
                changed = True

        return changed


class CustomBatchDenseLayer(Pattern):
    def __init__(self, target_mod, constraint_fn, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod
        self.constraint_fn = constraint_fn

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        return False
        if not hasattr(graph_module, "custom_batch_dense_layer_replacement"):
            graph_module.add_submodule("custom_batch_dense_layer_replacement", self.target_mod())
        for node in reversed(graph_module.graph.nodes):
            is_match, bmm_params = match_bmm_act(node)
            if is_match and self.constraint_fn(*fx.map_arg(bmm_params, lambda x: x.meta["val"])):
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "custom_batch_dense_layer_replacement", args=tuple(bmm_params)
                    )
                node.replace_all_uses_with(new_node)
                act_str = bmm_params[-1]
                if act_str != "none":
                    before_act = node.args[0]
                    graph_module.graph.erase_node(node)
                    graph_module.graph.erase_node(before_act)
                else:
                    graph_module.graph.erase_node(node)
                changed = True
        return changed
