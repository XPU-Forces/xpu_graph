from typing import Optional

from torch import fx
from torch.fx import map_arg

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DenseParams


def _is_serial_mm_2dot(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if (
        (node.target != "fused_matmul_replacement")
        and (node.target != "fused_matmul_add_replacement")
        and (node.target != "fused_matmul_act_replacement")
        and (node.target != "fused_matmul_add_act_replacement")
    ):
        return False, []

    dense_params1 = DenseParams().set_params(*node.args)

    dense_result0 = dense_params1.input
    if (
        (dense_result0.target != "fused_matmul_replacement")
        and (dense_result0.target != "fused_matmul_add_replacement")
        and (dense_result0.target != "fused_matmul_act_replacement")
        and (dense_result0.target != "fused_matmul_add_act_replacement")
    ):
        return False, []

    if len(dense_result0.users) > 1:
        return False, []

    dense_params0 = DenseParams().set_params(*dense_result0.args)

    is_up_bias, is_down_bias = False, False
    if dense_params0.bias is not None:
        is_up_bias = True
    if dense_params1.bias is not None:
        is_down_bias = True

    is_up_act, is_down_act = True, True
    if dense_params0.act == "none":
        is_up_act = False
    elif dense_params0.act == "relu":
        is_up_act = True
    else:
        return False, []

    if dense_params1.act == "none":
        is_down_act = False
    elif dense_params1.act == "relu":
        is_down_act = True
    else:
        return False, []

    if dense_params0.weight_trans or dense_params0.weight_trans:
        return False, []

    return True, [
        dense_params0.input,
        dense_params0.weight,
        dense_params0.bias,
        dense_params1.weight,
        dense_params1.bias,
        dense_params0.act,
        dense_params1.act,
        dense_params0.weight_trans,
        dense_params1.weight_trans,
        is_up_bias,
        is_down_bias,
        is_up_act,
        is_down_act,
    ]


def _is_serial_mm_3dot(
    node: fx.Node,
) -> tuple[bool, Optional[fx.Node]]:
    if node.target != "fused_dense_tower_2_replacement":
        return False, []

    matmul_node = node.args[0]
    if (
        (matmul_node.target != "fused_matmul_replacement")
        and (matmul_node.target != "fused_matmul_add_replacement")
        and (matmul_node.target != "fused_matmul_act_replacement")
        and (matmul_node.target != "fused_matmul_add_act_replacement")
    ):
        return False, []

    if len(matmul_node.users) > 1:
        return False, []

    dense_params0 = DenseParams().set_params(*matmul_node.args)

    if dense_params0.weight_trans:
        return False, []

    is_first_bias = False
    if dense_params0.bias is not None:
        is_first_bias = True

    is_first_act = True
    if dense_params0.act == "none":
        is_first_act = False
    elif dense_params0.act == "relu":
        is_first_act = True
    else:
        return False, []

    ffn_params = [
        dense_params0.input,
        dense_params0.weight,
        dense_params0.bias,
    ]
    ffn_params += node.args[1:5]
    ffn_params += [dense_params0.act]
    ffn_params += node.args[5:7]
    ffn_params += [dense_params0.weight_trans]
    ffn_params += node.args[7:9]
    ffn_params += [is_first_bias]
    ffn_params += node.args[9:11]
    ffn_params += [is_first_act]
    ffn_params += node.args[11:]

    return True, ffn_params


class FusedDenseTower2(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, constraint_fn, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod
        self.constraint_fn = constraint_fn

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_dense_tower_2_replacement"):
            graph_module.add_submodule("fused_dense_tower_2_replacement", self.target_mod())
        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_serial_mm_2dot(node)
            if is_match and self.constraint_fn(*map_arg(tinyffn_param, lambda x: x.meta["val"])):
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_dense_tower_2_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified


class FusedDenseTower3(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, constraint_fn, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod
        self.constraint_fn = constraint_fn

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fused_dense_tower_3_replacement"):
            graph_module.add_submodule("fused_dense_tower_3_replacement", self.target_mod())

        for node in reversed(graph_module.graph.nodes):
            is_match, tinyffn_param = _is_serial_mm_3dot(node)
            if is_match and self.constraint_fn(*map_arg(tinyffn_param, lambda x: x.meta["val"])):
                with graph_module.graph.inserting_before(node):
                    new_node = graph_module.graph.call_module(
                        "fused_dense_tower_3_replacement",
                        args=(tuple(tinyffn_param)),
                    )
                node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(node)
                is_modified = True

        return is_modified
