from .compiler import XpuGraph, optimize_graph
from .config import Target, OptLevel, XpuGraphConfig
from .cache import XpuGraphCache, default_cache
from typing import Dict, Any, Optional

__all__ = [
    "XpuGraph",
    "XpuGraphConfig",
    "Target",
    "OptLevel",
    "optimize_graph",
    "XpuGraphCache",
    "default_cache",
    "mlu_compiler",
]


def mlu_compiler(
    freeze: bool = True,
    opt_level: OptLevel = OptLevel.level1,
    constant_folding: bool = True,
    cache: Optional[XpuGraphCache] = None,
    debug: bool = False,
    vendor_compiler_config: Dict[str, Any] = None,
) -> XpuGraph:
    """
    Create an MLU compiler configuration and return an XpuGraph instance.

    Args:
        freeze: Whether to freeze the graph.
        opt_level: Optimization level.
        constant_folding: Whether to enable constant folding.
        cache: Cache for compiled graphs. Uses default cache if None.
        debug: Whether to enable debug mode.
        vendor_compiler_config: Additional vendor-specific compiler configuration.

    Returns:
        An XpuGraph instance configured for MLU.
    """
    if vendor_compiler_config is None:
        vendor_compiler_config = {"mode": "reduce-overhead"}

    config = XpuGraphConfig(
        target=Target.mlu,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
        debug=debug,
        vendor_compiler_config=vendor_compiler_config,
    )
    return XpuGraph(config, cache)
