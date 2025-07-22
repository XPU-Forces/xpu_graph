from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import (
    BatchDenseLayer,
    DenseLayer,
)


class CustomDenseLayer(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        fast_act = True if self._opt_level == OptLevel.level3 else False
        if not hasattr(graph_module, "custom_dense_layer_replacement"):
            graph_module.add_submodule("custom_dense_layer_replacement", self.target_mod(fast_act))
        changed = False
        for node in reversed(graph_module.graph.nodes):
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), DenseLayer):
                node.target = "custom_dense_layer_replacement"
                changed = True
        return changed


class CustomBatchDenseLayer(Pattern):
    def __init__(self, target_mod, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        if not hasattr(graph_module, "custom_batch_dense_layer_replacement"):
            graph_module.add_submodule("custom_batch_dense_layer_replacement", self.target_mod())
        for node in reversed(graph_module.graph.nodes):
            if node.op == "call_module" and isinstance(getattr(graph_module, node.target), BatchDenseLayer):
                node.target = "custom_batch_dense_layer_replacement"
                changed = True
        return changed
