import importlib
import pkgutil

from xpu_graph.config import XpuGraphConfig
from xpu_graph.passes.patterns.pattern import AutoMatchPattern, Pattern, PatternGroup


def get_all_patterns(config: XpuGraphConfig):
    patterns = {
        PatternGroup.GROUP0: [],
        PatternGroup.GROUP1: [],
        PatternGroup.GROUP2: [],
    }

    using_ge_backend_with_super_kernel = (
        config.vendor_compiler_config is not None
        and config.vendor_compiler_config.get("compiler", None) == "ge"
        and config.vendor_compiler_config.get("enable_super_kernel", False)
        and config.vendor_compiler_config.get("mode", None) == None
    )

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")

        for name in dir(module):
            pat = getattr(module, name)
            if (
                isinstance(pat, type)
                and issubclass(pat, Pattern)
                and pat.__module__.startswith(__name__)
                and pat not in (Pattern, AutoMatchPattern)
                and pat.filter(config)
            ):
                # NOTE(liuyuan): The nodes for super kernel may have side-effects on non-GE backend.
                if pat.__name__ == "ScopedSuperKernel" and not using_ge_backend_with_super_kernel:
                    continue
                else:
                    patterns[pat._pattern_group].append(pat())
    return patterns
