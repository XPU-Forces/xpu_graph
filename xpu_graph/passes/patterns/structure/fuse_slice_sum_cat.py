import torch
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_cat_op, check_meta_2d, check_slice_op, check_sum_op
from ..utils.default_replacements import DefaultSliceSumCatModule
from ..utils.submodule_manager import register_new_submodule


# This function identifies slice->sum patterns in a computation graph (gm).
# The detected pattern consists of:
#  - A slice operation (input: 3D, output: 3D)
#  - A sum operation (input: 3D, output: 2D)
# Both slice and sum nodes should have only one output to maintain a strict computational structure.
def find_slice_sum_pattern(gm):
    # Dictionary to store mapping from source nodes to sum nodes
    slice_dict = {}
    for node in gm.graph.nodes:
        ### Identify sum operation: input 3D, output 2D ###
        if not check_sum_op(node):
            continue
        if not check_meta_2d(node):
            continue
        # check sum dim
        if node.args[1] != [1]:
            continue
        # don't keep dim
        if len(node.args) > 2:
            continue
        if len(node.users) != 1:
            continue

        ### Identify slice operation: input 3D, output 3D ###
        slice_node = node.args[0]
        if not check_slice_op(slice_node):
            continue
        # check slice dim
        if slice_node.args[1] != 1:
            continue
        # This restriction is currently in place but will be removed in the future.
        if slice_node.args[2] != 0:
            continue
        if len(slice_node.users) != 1:
            continue

        src_node = slice_node.args[0]
        if src_node not in slice_dict:
            slice_dict[src_node] = []
        slice_dict[src_node].append(node)
    return slice_dict


def match_slice_dict(slice_dict, node):
    for src_node, s_list in slice_dict.items():
        if node in s_list:
            return True, src_node, s_list
    return False, None, None


def match_slice_sum_cat_pattern(cat_input, slice_dict):
    for idx in range(len(cat_input)):
        s = cat_input[idx]
        if not check_sum_op(s):
            continue
        is_match, src_node, s_list = match_slice_dict(slice_dict, s)
        if not is_match:
            continue
        # Identify a contiguous sequence of sum nodes
        sum_start = idx
        sum_end = idx
        # Search for sum nodes that are derived from the same slice operation
        for next_idx in range(idx + 1, len(cat_input)):
            next_s = cat_input[next_idx]
            if next_s in s_list:
                sum_end = next_idx
            else:
                break
        # Ensure there are at least two consecutive sum nodes
        if (sum_end - sum_start) < 2:
            continue
        match_sum_list = cat_input[sum_start : sum_end + 1]
        # Record the slice parameters from the matched sum nodes
        args = [(s.args[0].args[2], s.args[0].args[3]) for s in match_sum_list]
        return True, (sum_start, sum_end), src_node, args
    return False, None, None, None


class FusedSliceSumCat(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, constraint_fn, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod
        self.constraint_fn = constraint_fn

    def process(self, gm: fx.GraphModule):
        changed = False
        changed1 = True
        while changed1:
            changed1 = self.find_slice_sum_cat(gm)
            changed = changed | changed1

        return changed

    def find_slice_sum_cat(self, gm: fx.GraphModule):
        changed = False
        slice_dict = find_slice_sum_pattern(gm)
        if slice_dict == {}:
            return changed
        # slice->sum->cat
        for node in gm.graph.nodes:
            is_cat, _ = check_cat_op(node)
            if not is_cat:
                continue
            ori_cat_input = node.args[0]
            is_match, range_, src_node, slice_params = match_slice_sum_cat_pattern(ori_cat_input, slice_dict)
            if not is_match:
                continue
            with gm.graph.inserting_before(node):
                if self.constraint_fn(src_node.meta["val"], slice_params):
                    module_name = register_new_submodule(
                        gm,
                        "fused_slice_sum_cat",
                        self.target_mod,
                        args=(slice_params,),
                    )
                else:
                    # use default module for not-fitted cases to save from redundant pattern matching
                    module_name = register_new_submodule(
                        gm,
                        "default_slice_sum_cat",
                        DefaultSliceSumCatModule,
                        args=(slice_params,),
                    )
                new_node = gm.graph.call_module(
                    module_name,
                    args=(src_node,),
                )
                new_cat_input = ori_cat_input[: range_[0]] + [new_node] + ori_cat_input[range_[1] + 1 :]
                concat_node = gm.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(new_cat_input, -1),
                    name=node.name + "_1",
                )

            node.replace_all_uses_with(concat_node)
            gm.graph.erase_node(node)
            match_sum_list = ori_cat_input[range_[0] : range_[1] + 1]
            for s in match_sum_list:
                if len(s.users) == 0:
                    gm.graph.erase_node(s)
            changed = True
        return changed
