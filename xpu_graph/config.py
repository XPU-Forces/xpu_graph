from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering


class Target(Enum):
    ascend = "ascend"
    mlu = "mlu"
    none = "none"


@total_ordering
class OptLevel(Enum):
    level0 = 0
    level1 = 1
    level2 = 2
    level3 = 3

    def __lt__(self, other):
        if isinstance(other, OptLevel):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, OptLevel):
            return self.value == other.value
        return NotImplemented


class ExecuteMode(Enum):
    eager = "eager"
    graph = "graph"


@dataclass
class XpuGraphConfig:
    """Configuration for XPU graph execution."""

    debug: bool = False
    target: Target = field(default=Target.none)
    opt_level: OptLevel = OptLevel.level1
    is_training: bool = True
    dump_graph: bool = False
    enable_cache: bool = True
    use_xpu_ops: bool = False  # Use xpu_ops or not
    freeze: bool = False  # Freeze parameter, will do better constant_folding
    constant_folding: bool = False
    skip_all_pass: bool = (
        False  # Default false, use for debug, which will skip all passes of xpu_ops
    )

    # Mode: {"cudagraphs", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
    # we add a "cudagraphs" option. At this mode, only torch.compile in-tree backend cudugraphs will be enable.
    # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
    vendor_compiler_config: Optional[Dict[str, Any]] = None
