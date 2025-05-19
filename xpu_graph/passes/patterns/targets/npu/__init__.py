import pkgutil
import importlib
import os

from xpu_graph.passes.patterns.pattern import Pattern, AutoMatchPattern, PatternGroup
from xpu_graph.config import XpuGraphConfig
from xpu_graph.utils import logger


def get_all_patterns(config: XpuGraphConfig):
    patterns = {
        PatternGroup.GROUP0: [],
        PatternGroup.GROUP1: [],
        PatternGroup.GROUP2: [],
    }

    if config.export_mode:
        logger.warning(
            "AOTI on Ascend NPU do not support DIY triton kernel, npu patterns will be ignored!"
        )
        return patterns

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        for name in dir(module):
            pat = getattr(module, name)
            if (
                isinstance(pat, type)
                and issubclass(pat, Pattern)
                and pat not in (Pattern, AutoMatchPattern)
                and pat._opt_level <= config.opt_level
            ):
                patterns[pat._pattern_group].append(pat())
    return patterns
