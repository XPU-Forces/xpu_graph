import torch.fx as fx
from torch.fx.node import map_arg

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import (
    DefaultSDPA,
)


class FlashAttention(Pattern):
    def __init__(self, target_mod, constraint_fn, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod
        self.constraint_fn = constraint_fn

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        if not hasattr(graph_module, "fa_replacement"):
            graph_module.add_submodule("fa_replacement", self.target_mod())
        for node in reversed(graph_module.graph.nodes):
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), DefaultSDPA):
                if self.constraint_fn(*map_arg(node.args, lambda arg: arg.meta["val"])):
                    node.target = "fa_replacement"
                    changed = True
        return changed
