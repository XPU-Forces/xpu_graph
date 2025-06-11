from typing import Callable, List, overload

import torch
import torch.fx as fx

from xpu_graph.config import OptLevel, Target, XpuGraphConfig
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.utils import logger

from .pattern import Pattern, PatternGroup


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
            logger.debug(f"xpu_graph enable builtin {group} patterns: {[str(pat) for pat in group_patterns]}")

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
    def register_pattern(self, pattern: Pattern):
        ...

    @overload
    def register_pattern(self, matcher: Callable, replacement: Callable):
        ...

    def register_pattern(self, *args, **kwargs):
        if len(args) == 1:
            pat = args[0]
            if pat._opt_level <= self._opt_level:
                self._patterns[pat._pattern_group].append(pat)
                pat._set_level(self._opt_level)
            else:
                logger.warning(
                    f"xpu_graph skip pattern {pat} with opt_level {pat._opt_level} and current opt_level is {self._opt_level}"
                )
        elif len(args) == 2:

            class _Pattern(Pattern):
                def __init__(self):
                    super().__init__()

                def __call__(self, gm: fx.GraphModule) -> bool:
                    from torch.fx import subgraph_rewriter

                    match = subgraph_rewriter.replace_pattern(gm, args[0], args[1])

                    return len(match)

            if "supported_stages" in kwargs:
                _Pattern._support_stages = kwargs["supported_stages"]
            if "opt_level" in kwargs:
                _Pattern._opt_level = kwargs["opt_level"]
            pat = _Pattern()
            if pat._opt_level <= self._opt_level:
                self._patterns[pat._pattern_group].append(pat)
                pat._set_level(self._opt_level)
            else:
                logger.warning(
                    f"xpu_graph skip pattern {pat} with opt_level {pat._opt_level} and current opt_level is {self._opt_level}"
                )

    def insert_patterns(self, patterns: List[Pattern]):
        tmp = {
            PatternGroup.GROUP0: [],
            PatternGroup.GROUP1: [],
            PatternGroup.GROUP2: [],
        }

        for pattern in patterns:
            tmp[pattern._pattern_group].append(pattern)

        for group in self._patterns.keys():
            tmp[group].extend(self._patterns[group])
            self._patterns[group] = tmp[group]

    def _set_level(self, level: OptLevel):
        self._opt_level = level
        for group in self._patterns.keys():
            for pattern in self._patterns[group]:
                pattern._set_level(level)
