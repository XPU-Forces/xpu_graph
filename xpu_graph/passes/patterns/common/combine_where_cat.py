import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ..utils.check_ops import check_where_op, check_zeros_op


def match_samezeromask(target_node, cond):
    if not check_where_op(target_node):
        return False
    # currently, the combination rule is strict:
    # only when the cond is same and the mask_val is zero
    # and the original node is not broadcasted
    return (
        check_zeros_op(target_node.args[1])
        and (
            isinstance(target_node.args[2], fx.Node)
            and target_node.meta["val"].shape == target_node.args[2].meta["val"].shape
        )
        and (cond is None or cond == target_node.args[0])
    )


class CombineWhereCat(Pattern):
    _opt_level = OptLevel.level1
    """
    input_1 -> where_1 ----\
    input_2 -> where_2 ------> cat -> output
    input_3 -> where_3 ----/
    other   ---------------/
    ---->
    input_1 ----\
    input_2 -----> cat -> where -> cat -> output
    input_3 ----/              /
    other   ------------------/

    The mask should be same
    The other arg should be const
    N*where + 1*cat -> 1*cat + 1+where
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        MINI_LEN = 3

        for node in reversed(graph_module.graph.nodes):
            if node.op != "call_function" or (
                node.target != torch.ops.aten.cat.default and node.target != torch.ops.aten.stack.default
            ):
                continue
            if len(node.args[0]) < MINI_LEN:
                continue
            is_stack = node.target == torch.ops.aten.stack.default

            ori_cat_input = node.args[0]

            if "dim" in node.kwargs:
                cat_axis = node.kwargs["dim"]
            elif len(node.args) > 1:
                cat_axis = node.args[1]
            else:
                cat_axis = 0

            if cat_axis < 0:
                cat_axis += len(ori_cat_input[0].meta["val"].shape)

            replaced_map = {}

            idx_r = 0
            for idx, start_arg in enumerate(ori_cat_input):
                if idx < idx_r:
                    # already tested or combined in previous node
                    continue
                if match_samezeromask(start_arg, None):
                    cond = start_arg.args[0]
                    idx_r = idx + 1
                    while idx_r < len(ori_cat_input) and match_samezeromask(ori_cat_input[idx_r], cond):
                        idx_r += 1
                    if idx_r - idx < MINI_LEN:
                        continue
                    n_list = ori_cat_input[idx:idx_r]

                    combined_inputs = [n.args[2] for n in n_list]
                    with graph_module.graph.inserting_before(node):
                        combined_input_node = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.cat.default,
                            args=(combined_inputs, cat_axis),
                            name=node.name + "_combined_where_inputs",
                        )

                        extra_zeros_kwargs = dict(start_arg.kwargs)
                        zeros_node = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.zeros_like.default,
                            args=(combined_input_node,),
                            name=node.name + "_combined_where_zeros",
                            kwargs=extra_zeros_kwargs,
                        )

                        cond_shape = cond.meta["val"].shape
                        cond_repeated_dim = cat_axis - len(start_arg.meta["val"].shape) + len(cond_shape)
                        if cond_repeated_dim >= 0 and cond_shape[cond_repeated_dim] != 1:
                            # if the cond's dim at cat_axis is not 1, it is sure that all inputs should be the same shape
                            repeats = [1] * len(cond_shape)
                            repeats[cond_repeated_dim] = len(combined_inputs)
                            cond = graph_module.graph.create_node(
                                op="call_function",
                                target=torch.ops.aten.repeat.default,
                                args=(cond, repeats),
                                name=node.name + "_combined_where_repeated_condition",
                            )
                        where_node = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.where.self,
                            args=(cond, zeros_node, combined_input_node),
                            name=node.name + "_combined_where_replacement",
                        )
                        replaced_map[(idx, idx_r)] = where_node
            if len(replaced_map) == 0:
                continue
            previous = 0
            new_cat_input = []
            for idx, idx_r in replaced_map:
                new_cat_input = new_cat_input + ori_cat_input[previous:idx] + [replaced_map[(idx, idx_r)]]
                previous = idx_r
            new_cat_input = new_cat_input + ori_cat_input[previous:]

            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    replaced_cat = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, cat_axis),
                        name=node.name + "_combined_where_cat_replacement",
                    )
            else:
                replaced_cat = new_cat_input[0]

            if is_stack:
                with graph_module.graph.inserting_before(node):
                    replaced_cat = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.unflatten.int,
                        args=(replaced_cat, cat_axis, [len(ori_cat_input), -1]),
                        name=node.name + "_combined_where_stack_replacement",
                    )
            node.replace_all_uses_with(replaced_cat)
            changed = True
        return changed
