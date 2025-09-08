import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_op, is_exclusively_used


def match_same_arg(node, target_node, is_left):
    same_arg = 0 if is_left else 1
    diff_arg = 1 - same_arg
    # currently, the combination rule is strict:
    # only when the original node is not broadcasted and has same requires_grad
    return (
        check_op(node, target_node.target)
        and node.args[same_arg] == target_node.args[same_arg]
        and isinstance(node.args[diff_arg], fx.Node)
        and isinstance(target_node.args[diff_arg], fx.Node)
        and node.meta["val"].shape == node.args[diff_arg].meta["val"].shape
        and node.args[diff_arg].meta["val"].requires_grad == target_node.args[diff_arg].meta["val"].requires_grad
    )


def is_candidate_target(node):
    candidate_targets = [torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor]
    if node.op != "call_function" or node.target not in candidate_targets:
        return False
    return node.target in candidate_targets


class CombinePointwiseCat(Pattern):
    _opt_level = OptLevel.level1
    """
    input_1 -> poi_1 ----\
    input_2 -> poi_2 ------> cat -> output
    input_3 -> poi_3 ----/
    other   ---------------/
    ---->
    input_1 ----\
    input_2 -----> cat -> poi -> cat -> output
    input_3 ----/              /
    other   ------------------/

    The mask should be same
    The other arg should be const
    N*poi + 1*cat -> 1*cat + 1+poi
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
                if idx + 1 >= len(ori_cat_input):
                    break

                if is_candidate_target(start_arg) and is_exclusively_used(start_arg, node):
                    idx_r = idx + 1
                    is_left_arg_same = True
                    is_right_arg_same = True
                    while idx_r < len(ori_cat_input):
                        cur_left_matched = is_left_arg_same and match_same_arg(ori_cat_input[idx_r], start_arg, True)
                        cur_right_matched = is_right_arg_same and match_same_arg(ori_cat_input[idx_r], start_arg, False)
                        if not (cur_left_matched or cur_right_matched):
                            break
                        if not is_exclusively_used(ori_cat_input[idx_r], node):
                            break
                        is_left_arg_same = cur_left_matched
                        is_right_arg_same = cur_right_matched
                        idx_r += 1
                    if idx_r - idx < MINI_LEN:
                        continue
                    n_list = ori_cat_input[idx:idx_r]

                    combined_inputs = [n.args[1 if is_left_arg_same else 0] for n in n_list]
                    shared_input = start_arg.args[0 if is_left_arg_same else 1]
                    with graph_module.graph.inserting_before(node):
                        combined_input_node = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.cat.default,
                            args=(combined_inputs, cat_axis),
                            name=node.name + "_combined_inputs",
                        )

                        shared_shape = shared_input.meta["val"].shape
                        shared_repeated_dim = cat_axis - len(start_arg.meta["val"].shape) + len(shared_shape)
                        if shared_repeated_dim >= 0 and shared_shape[shared_repeated_dim] != 1:
                            # if the cond's dim at cat_axis is not 1, it is sure that all inputs should be the same shape
                            repeats = [1] * len(shared_shape)
                            repeats[shared_repeated_dim] = len(combined_inputs)
                            shared_input = graph_module.graph.create_node(
                                op="call_function",
                                target=torch.ops.aten.repeat.default,
                                args=(shared_input, repeats),
                                name=node.name + "_shared_repeats",
                            )

                        combined_output = graph_module.graph.create_node(
                            op="call_function",
                            target=start_arg.target,
                            args=(
                                (shared_input, combined_input_node)
                                if is_left_arg_same
                                else (combined_input_node, shared_input)
                            ),
                            name=node.name + "_combined_replacement",
                        )
                        replaced_map[(idx, idx_r)] = combined_output
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
                        name=node.name + "_combined_cat_replacement",
                    )
            else:
                replaced_cat = new_cat_input[0]

            if is_stack:
                with graph_module.graph.inserting_before(node):
                    replaced_cat = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.unflatten.int,
                        args=(replaced_cat, cat_axis, [len(ori_cat_input), -1]),
                        name=node.name + "_combined_stack_replacement",
                    )
            node.replace_all_uses_with(replaced_cat)
            changed = True
        return changed
