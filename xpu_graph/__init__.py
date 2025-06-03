import dataclasses
from typing import Any, Dict

from .cache import XpuGraphCache, XpuGraphLocalCache, default_cache, no_cache
from .compiler import XpuGraph, optimize_graph
from .config import OptLevel, Target, XpuGraphConfig
from .passes.patterns.plugin_pattern import *
from .version import __version__

__all__ = [
    "XpuGraph",
    "XpuGraphConfig",
    "Target",
    "OptLevel",
    "optimize_graph",
    "XpuGraphCache",
    "default_cache",
    "mlu_compiler",
    "ascend_compiler",
    "register_plugin_pattern",
    "register_this_as_plugin_pattern",
    "register_this_as_pattern_constraint",
    "deregister_plugin_patterns",
    "enable_plugin_patterns",
]


def ascend_compiler(
    freeze: bool = False,
    opt_level: OptLevel = OptLevel.level1,
    constant_folding: bool = True,
    cache: XpuGraphCache = default_cache(),
    debug: bool = False,
    vendor_compiler_config: Dict[str, Any] = {"mode": "default"},
):
    config = XpuGraphConfig(
        is_training=False,
        target=Target.ascend,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
        debug=debug,
        vendor_compiler_config=vendor_compiler_config,
    )
    return XpuGraph(config, cache)


def npu_compiler(
    freeze: bool = False,
    opt_level: OptLevel = OptLevel.level1,
    constant_folding: bool = True,
    cache: XpuGraphCache = default_cache(),
    debug: bool = False,
    vendor_compiler_config: Dict[str, Any] = {"mode": "default"},
):
    config = XpuGraphConfig(
        is_training=False,
        target=Target.npu,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
        debug=debug,
        vendor_compiler_config=vendor_compiler_config,
    )
    return XpuGraph(config, cache)


def mlu_compiler(
    is_training: bool,
    **patch_configs,
) -> XpuGraph:
    """
    Create an MLU compiler configuration and return an XpuGraph instance.

    Possible Patch Args:
        freeze: Whether to freeze the graph.
        opt_level: Optimization level.
        constant_folding: Whether to enable constant folding.
        cache: Cache for compiled graphs. Uses default cache if None.
        debug: Whether to enable debug mode.
        vendor_compiler_config: Additional vendor-specific compiler configuration.

    Returns:
        An XpuGraph instance configured for MLU.
    """
    assert "target" not in patch_configs, "target is not allowed to be set here"

    if "cache" not in patch_configs:
        cache = no_cache() if is_training else default_cache()
    elif isinstance(patch_configs["cache"], str):
        cache = XpuGraphLocalCache(patch_configs["cache"])
    else:
        assert isinstance(patch_configs["cache"], XpuGraphCache), "cache must be a XpuGraphCache instance"
        cache = patch_configs["cache"]

    default_config = _MLU_TRAIN_CONFIG if is_training else _MLU_INFER_CONFIG

    if "opt_level" in patch_configs and not isinstance(patch_configs["opt_level"], OptLevel):
        patch_configs["opt_level"] = OptLevel(int(patch_configs["opt_level"]))
    patch_configs = {
        field.name: patch_configs[field.name]
        for field in dataclasses.fields(XpuGraphConfig)
        if field.name in patch_configs
    }
    config = dataclasses.replace(default_config, **patch_configs)

    if not is_training:
        import torch_mlu_ops
    return XpuGraph(config, cache)


_MLU_TRAIN_CONFIG = XpuGraphConfig(
    is_training=True,
    debug=False,
    target=Target.mlu,
    enable_cache=True,
    freeze=False,
    opt_level=OptLevel.level2,
    constant_folding=False,
    vendor_compiler_config={"mode": "default"},
)

_MLU_INFER_CONFIG = XpuGraphConfig(
    is_training=False,
    debug=False,
    target=Target.mlu,
    enable_cache=True,
    freeze=False,
    opt_level=OptLevel.level2,
    constant_folding=False,
    vendor_compiler_config={"mode": "reduce-overhead"},
)
