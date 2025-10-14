import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import AutoMatchPattern


class Gelu(AutoMatchPattern):
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def rewriter(self, gm: fx.GraphModule, rule_name: str, node_map: dict) -> bool:
        assert len(node_map) == 5 or len(node_map) == 8
        if len(node_map) == 5:
            div_node = node_map["div"]
            add_node = node_map["add"]
            mul_node = node_map["mul"]
            mul2_node = node_map["mul2"]

            import math

            if div_node.args[1] != math.sqrt(2):
                return False
            if add_node.args[1] != 1 and add_node.args[0] != 1:
                return False
            if mul_node.args[1] != 0.5 and mul_node.args[0] != 0.5:
                return False

            with gm.graph.inserting_before(div_node):
                gelu_node = gm.graph.call_function(torch.ops.aten.gelu.default, args=(div_node.args[0],))

            mul2_node.replace_all_uses_with(gelu_node)
        else:
            mul2_node = node_map["mul2"]
            pow_node = node_map["pow"]
            mul3_node = node_map["mul3"]
            add2_node = node_map["add2"]
            mul4_node = node_map["mul4"]
            tanh_node = node_map["tanh"]
            add3_node = node_map["add3"]
            mul5_node = node_map["mul5"]

            import math

            if mul2_node.args[1] != 0.5:
                return False
            if pow_node.args[1] != 3.0:
                return False
            if mul3_node.args[1] != 0.044715:
                return False
            if mul4_node.args[1] != math.sqrt(2.0 / math.pi):
                return False
            if add3_node.args[1] != 1.0:
                return False
            with gm.graph.inserting_before(mul2_node):
                gelu_node = gm.graph.call_function(torch.ops.aten.gelu.default, args=(mul2_node.args[0],))
            mul5_node.replace_all_uses_with(gelu_node)

        return True
