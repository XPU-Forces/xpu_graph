from typing import Optional, Tuple, Union

import torch
import torch_mlu
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger

from ...utils.check_ops import get_shape

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node


"""
    sample_code:
    %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze, [1, 1, 256]), kwargs = {})
    %gather : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%arg0_1, 1, %repeat), kwargs = {})
"""


class FusedGatherToCopy(Pattern):
    _opt_level = OptLevel.level2
    # _support_stages = [FxStage.inference, FxStage.pregrad]
    """
    a: [b, 1, k] b: [b, k, 1]
    torch.bmm(a,b) -> torch.sum(a.squeeze(1) * b.squeeze(-1), dim=-1).unsqueeze(-1)
    """

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and node.target in [torch.ops.aten.mm.default, torch.ops.aten.matmul.default, torch.ops.aten.bmm.default]
        ]
        for mm_node in candidates:
            left_input = mm_node.args[0]
            right_input = mm_node.args[1]
            shape_l = get_shape(left_input)
            shape_r = get_shape(right_input)
            if shape_l is None or shape_r is None:
                continue
            m = shape_l[-2]
            k = shape_l[-1]
            n = shape_r[-1]
            with graph_module.graph.inserting_before(mm_node):
                if m == 1 and n == 1:
                    squeeze_l = graph_module.graph.call_function(torch.ops.aten.squeeze.dim, args=(left_input, -2))
                    squeeze_r = graph_module.graph.call_function(torch.ops.aten.squeeze.dim, args=(right_input, -1))
                elif k == 1:
                    squeeze_l = graph_module.graph.call_function(torch.ops.aten.squeeze.dim, args=(left_input, -1))
                    squeeze_r = graph_module.graph.call_function(torch.ops.aten.squeeze.dim, args=(right_input, -2))
                else:
                    continue
                mul_node = graph_module.graph.call_function(torch.ops.aten.mul.Tensor, args=(squeeze_l, squeeze_r))
                sum_node = graph_module.graph.call_function(torch.ops.aten.sum.dim_IntList, args=(mul_node, [-1]))
                unsqueeze1 = graph_module.graph.call_function(torch.ops.aten.unsqueeze.default, args=(sum_node, -1))
                new_node = graph_module.graph.call_function(torch.ops.aten.unsqueeze.default, args=(unsqueeze1, -1))
                mm_node.replace_all_uses_with(new_node)
                graph_module.graph.erase_node(mm_node)
                is_modified = True
        return is_modified
