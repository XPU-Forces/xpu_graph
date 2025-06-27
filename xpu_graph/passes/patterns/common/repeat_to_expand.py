from typing import Optional, Tuple, Union

import torch
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ..utils.check_ops import check_op

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node


"""
    sample_code:
    %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze, [1, 1, 256]), kwargs = {})
    %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg0_1, 1, %repeat), kwargs = {})
"""


class ChangeRepeatToExpand(Pattern):
    _opt_level = OptLevel.level1
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        candidates = [node for node in graph_module.graph.nodes if check_op(node, torch.ops.aten.gather.default)]
        for gather_node in candidates:
            repeat_node = gather_node.args[2]
            gather_dim = gather_node.args[1]
            if repeat_node.target != torch.ops.aten.repeat.default:
                continue
            if len(repeat_node.users) != 1:
                continue

            repeat_param = repeat_node.args[1]
            expand_param = [-1 if i == 1 else i for i in repeat_param]

            with graph_module.graph.inserting_before(repeat_node):
                new_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.expand.default,
                    args=(repeat_node.args[0], expand_param),
                    name=gather_node.name + "_replacement",
                )
            repeat_node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(repeat_node)
            is_modified = True
        return is_modified
