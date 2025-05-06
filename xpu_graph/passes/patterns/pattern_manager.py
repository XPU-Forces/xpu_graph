from typing import Callable, overload, List

import torch
import torch.fx as fx

from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.config import XpuGraphConfig, Target
from xpu_graph.fx_utils import FxStage
from .pattern import Pattern, PatternGroup
from xpu_graph.utils import logger


class PatternManager(Optimizer):
    def __init__(self, config: XpuGraphConfig):
        super().__init__()

        self._patterns = {
            PatternGroup.GROUP0: [],
            PatternGroup.GROUP1: [],
            PatternGroup.GROUP2: [],
        }

        self._enable_patterns = {
            PatternGroup.GROUP0: [],
            PatternGroup.GROUP1: [],
            PatternGroup.GROUP2: [],
        }

        from .common import get_all_patterns as get_common_patterns

        for group, patterns in get_common_patterns(config).items():
            self._patterns[group] += patterns

        from .structure import get_all_patterns as get_structure_patterns

        for group, patterns in get_structure_patterns(config).items():
            self._patterns[group] += patterns

        if config.use_xpu_ops:
            from .xpu_ops import get_all_patterns as get_xpu_ops_patterns

            for group, patterns in get_xpu_ops_patterns(config).items():
                self._patterns[group] += patterns

        from .targets import get_all_patterns as get_target_patterns

        for group, patterns in get_target_patterns(config).items():
            self._patterns[group] += patterns

    def reset_patterns_with_stage(self, stage):
        self._enable_patterns = {
            PatternGroup.GROUP0: [],
            PatternGroup.GROUP1: [],
            PatternGroup.GROUP2: [],
        }
        for group in self._patterns.keys():
            for pattern in self._patterns[group]:
                if stage in pattern._support_stages:
                    pattern.set_current_stage(stage)
                    self._enable_patterns[group].append(pattern)

        for group, group_patterns in self._enable_patterns.items():
            logger.debug(
                lambda:f"xpu_graph enable builtin {group} patterns: {[pat.__class__.__name__ for pat in group_patterns]}"
            )

    def get_pass_with_stage(self, stage):
        self.reset_patterns_with_stage(stage)
        return self

    def process(self, gm: fx.GraphModule) -> bool:
        changed = False
        loop_time = 5
        for group in sorted(self._patterns.keys()):
            for i in range(loop_time):
                for pattern in self._enable_patterns[group]:
                    changed = changed or pattern(gm)

        return changed

    @overload
    def register_pattern(self, pattern: Pattern): ...

    @overload
    def register_pattern(self, matcher: Callable, replacement: Callable): ...

    @overload
    def register_pattern(
        self, matcher: Callable, replacement: Callable, stage: FxStage
    ): ...

    def register_pattern(self, *args):
        if len(args) == 1:
            pat = args[0]
            self._patterns[pat._pattern_group].append(pat)
        elif len(args) == 2 or len(args) == 3:

            class _Pattern(Pattern):
                def __init__(self):
                    super().__init__()

                def __call__(self, gm: fx.GraphModule) -> bool:
                    from torch.fx import subgraph_rewriter

                    match = subgraph_rewriter.replace_pattern(gm, args[0], args[1])

                    return len(match)

            if len(args) == 3:
                _Pattern._support_stages = [args[2]]
            self._patterns[_Pattern._pattern_group].append(_Pattern())
