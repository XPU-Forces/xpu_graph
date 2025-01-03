from .compiler import XpuGraph
from .config import Target, OptLevel, XpuGraphConfig

__all__ = ["XpuGraph", "XpuGraphConfig", "Target", "OptLevel"]


def mlu_compiler(
    freeze: bool = True, opt_level: OptLevel = OptLevel.level1, graph: bool = True
):
    config = XpuGraphConfig(target=Target.mlu, freeze=freeze, opt_level=opt_level)
    if graph:
        config.vendor_compiler = {"mode": "reduce-overhead"}
    return XpuGraph(config)
