from torch import nn

from xpu_graph.config import Target
from xpu_graph.utils import logger


def get_all_patterns(config) -> dict:
    if config.target == Target.mlu:
        from .mlu import get_all_patterns

        return get_all_patterns(config)
    elif config.target == Target.npu:
        from .npu import get_all_patterns

        return get_all_patterns(config)
    return {}


def get_structure_replacements(config) -> dict:
    replacement_args = {}
    if config.target == Target.mlu:
        from .mlu.structure_replacements import get_structure_replacements

        for pat_name, args in get_structure_replacements(config).items():
            if isinstance(args, tuple):
                target_mod, constraint_fn = args
                replacement_args[pat_name] = {"target_mod": target_mod, "constraint_fn": constraint_fn}
            else:
                replacement_args[pat_name] = {"target_mod": args}

    return replacement_args
