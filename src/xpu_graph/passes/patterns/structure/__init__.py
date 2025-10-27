import importlib
import pkgutil

from xpu_graph.config import Target, XpuGraphConfig
from xpu_graph.passes.patterns.pattern import AutoMatchPattern, Pattern, PatternGroup
from xpu_graph.utils import logger


def get_all_patterns(config: XpuGraphConfig):
    patterns = {
        PatternGroup.GROUP0: [],
        PatternGroup.GROUP1: [],
        PatternGroup.GROUP2: [],
    }

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        from ..targets import get_structure_replacements

        structure_preplacements = get_structure_replacements(config)

        for name in dir(module):
            pat = getattr(module, name)
            if (
                isinstance(pat, type)
                and issubclass(pat, Pattern)
                and pat.__module__.startswith(__name__)
                and pat not in (Pattern, AutoMatchPattern)
                and pat._opt_level <= config.opt_level
            ):
                if pat.__name__ in structure_preplacements:
                    patterns[pat._pattern_group].append(pat(**structure_preplacements[pat.__name__]))

    return patterns
