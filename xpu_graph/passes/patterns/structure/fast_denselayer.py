from torch import fx

from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.default_replacements import (
    BatchDenseLayer,
    DenseLayer,
)


class FastDenseLayer(Pattern):
    _opt_level = OptLevel.level2

    def __init__(self, target_mod, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        fast_act = True if self._opt_level == OptLevel.level3 else False
        changed = False
        replaced_submods = [
            sub_name for sub_name, sub_mod in graph_module.named_modules() if isinstance(sub_mod, DenseLayer)
        ]
        for sub_name in replaced_submods:
            graph_module.delete_submodule(sub_name)
            graph_module.add_submodule(sub_name, self.target_mod(fast_act))
            changed = True
        return changed


class FastBatchDenseLayer(Pattern):
    def __init__(self, target_mod, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        replaced_submods = [
            sub_name for sub_name, sub_mod in graph_module.named_modules() if isinstance(sub_mod, BatchDenseLayer)
        ]
        for sub_name in replaced_submods:
            graph_module.delete_submodule(sub_name)
            graph_module.add_submodule(sub_name, self.target_mod())
            changed = True
        return changed
