import torch
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DefaultLayerNorm


class CustomLayerNorm(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference]

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "custom_layer_norm"):
            graph_module.add_submodule("custom_layer_norm", self.target_mod())
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), DefaultLayerNorm):
                inputs, weight, bias, eps = node.args
                if weight is None or bias is None:
                    continue
                with graph_module.graph.inserting_before(node):
                    custom_layernorm = graph_module.graph.call_module("custom_layer_norm", (inputs, weight, bias, eps))
                node.replace_all_uses_with(custom_layernorm, propagate_meta=True)
                is_modified = True
        return is_modified
