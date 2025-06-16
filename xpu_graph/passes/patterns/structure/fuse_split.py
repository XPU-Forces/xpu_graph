import operator

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern

from ..utils.check_ops import check_meta_2d

"""
    %split : call_function[target=torch.ops.aten.split.Tensor](args = (%addmm_35, 380, 1), kwargs = {})
    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%split, 0), kwargs = {})
    ...
    %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%split, 3), kwargs = {})
    ->
    %fused_slice_low : [num_users=1] = call_function[target=torch.ops.torch_mlu_triton.fused_slice_low.default](args = (%arg0_1, %_to_copy, 3), kwargs = {})
    %unbind : [num_users=4] = call_function[target=torch.ops.aten.unbind.int](args = (%fused_slice_low,), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%unbind, 0), kwargs = {})
    ...
    %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%unbind, 3), kwargs = {})
"""


class FusedSplit(Pattern):
    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, gm: fx.GraphModule):
        changed = False
        gm.add_submodule("fused_split", self.target_mod())
        candidates = [
            node for node in gm.graph.nodes if node.op == "call_function" and node.target == torch.ops.aten.split.Tensor
        ]
        for node in candidates:
            x = node.args[0]
            split_size = node.args[1]
            if len(node.args) > 2:
                dim = node.args[2]
            else:
                dim = 0
            if not check_meta_2d(x):
                continue
            if dim == 0:
                continue

            outputs = []
            index_nodes = []

            for user in list(node.users):
                if (
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and user.args[0] == node
                    and isinstance(user.args[1], int)
                ):
                    outputs.append(user.args[1])
                    index_nodes.append(user)

            if len(outputs) < 2:
                continue

            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_module("fused_split", args=(x, split_size, dim))
            for i, getitem_node in enumerate(index_nodes):
                with gm.graph.inserting_before(getitem_node):
                    out = gm.graph.call_function(operator.getitem, args=(new_node, outputs[i]))
                    getitem_node.replace_all_uses_with(out)
                    gm.graph.erase_node(getitem_node)

            gm.graph.erase_node(node)
            changed = True

        return changed
