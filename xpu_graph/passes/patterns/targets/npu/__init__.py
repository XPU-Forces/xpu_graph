import pkgutil
import importlib

from xpu_graph.passes.patterns.pattern import Pattern, AutoMatchPattern, PatternGroup
from xpu_graph.config import XpuGraphConfig
from xpu_graph.utils import logger


def get_all_patterns(config: XpuGraphConfig):
    patterns = {
        PatternGroup.GROUP0: [],
        PatternGroup.GROUP1: [],
        PatternGroup.GROUP2: [],
    }

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
                for stage in pat._stages:
                    patterns[pat._pattern_group].append(pat(stage))

    for group, group_patterns in patterns.items():
        logger.debug(f"xpu_graph enable builtin npu {group} patterns: {[pat.__class__.__name__ for pat in group_patterns]}")

    return patterns
