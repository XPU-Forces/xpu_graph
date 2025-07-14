import torch
import torch.nn.functional as F
from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import DefaultLayerNorm


class FastLayerNorm(Pattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference]

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        if not hasattr(graph_module, "fast_layernorm"):
            graph_module.add_submodule("fast_layernorm", self.target_mod())
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), DefaultLayerNorm):
                inputs, weight, bias, eps = node.args
                # 只有当weight不为None时才使用快速实现
                if weight is None:
                    continue
                with graph_module.graph.inserting_before(node):
                    fast_layernorm = graph_module.graph.call_module("fast_layernorm", (inputs, weight, bias, eps))
                node.replace_all_uses_with(fast_layernorm, propagate_meta=True)
                is_modified = True
        return is_modified