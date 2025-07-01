import warnings
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import Any, Dict, Optional, Union


class Target(Enum):
    mlu = "mlu"
    none = "none"
    npu = "npu"


@total_ordering
class OptLevel(Enum):
    level0 = 0  # Close all optimizer
    level1 = 1  # Reture results with bitwise alignment
    level2 = 2  # No guarantee fot bitwise alignment
    level3 = 3  # Placeholder, same with level2 now

    def __lt__(self, other):
        if isinstance(other, OptLevel):
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, OptLevel):
            return self.value == other.value
        return NotImplemented


@dataclass
class XpuGraphConfig:
    """Configuration for XPU graph execution."""

    is_training: bool  # Must fill, if is_training is True, XpuGraph will work as a training compiler, otherwise a inference compiler
    debug: bool = False
    target: Target = field(default_factory=lambda: Target.none)  # Target hardware backend
    opt_level: OptLevel = field(default_factory=lambda: OptLevel.level1)
    dump_graph: bool = False
    enable_cache: bool = True
    use_xpu_ops: bool = False  # Use xpu_ops or not
    freeze: bool = (
        # Only take effects when "is_training" is False.
        # Freezing parameter will change model's parameter from inputs into attributes.
        # This may help XpuGraph do better constant folding.
        False
    )
    constant_folding: bool = True
    folding_freezed_params: bool = (
        # Only take effects whe freeze is True and constant_folding is True
        # When freeze is True, params exists as attributes in GraphModule.
        # If folding_freezed_params is True, XpuGraph will treat freezed parameters as constants and fold them
        # If folding_freezed_params is False, XpuGraph will not fold freezed parameters and allow parameter hot-swapping
        True
    )

    # So far we only support configure "mode", because we mainly use "Inductor" as a vendor's compiler.
    # mode must be one of {"cudagraphs", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"},
    # we add a "cudagraphs" option. At this mode, XpuGraph will only enable torch.compile in-tree backend "cudugraphs".
    # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
    vendor_compiler_config: Optional[Dict[str, Any]] = None

    def _reset_config_with_env(self):
        import os

        if os.getenv("XPUGRAPH_DEBUG") is not None:
            self.debug = os.getenv("XPUGRAPH_DEBUG", "0") == "1"  # 1: enable 0: disable

        opt_level_env = os.getenv("XPUGRAPH_OPT_LEVEL", str(self.opt_level.value))
        if opt_level_env == "0":
            self.opt_level = OptLevel.level0
        elif opt_level_env == "1":
            self.opt_level = OptLevel.level1
        elif opt_level_env == "2":
            self.opt_level = OptLevel.level2
        elif opt_level_env == "3":
            self.opt_level = OptLevel.level3
        else:
            warnings.warn("Illegal XPUGRAPH_OPT_LEVEL value, XPUGRAPH_OPT_LEVEL will not take effect.")

        vendor_compiler_mode = os.getenv("VENDOR_COMPILER_MODE", "Null")
        if vendor_compiler_mode != "Null":
            if vendor_compiler_mode == "none":
                self.vendor_compiler_config = None
            else:
                if vendor_compiler_mode not in (
                    "default",
                    "cudagraphs",
                    "reduce-overhead",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                ):
                    warnings.warn("Illegal VENDOR_COMPILER_MODE value, VENDOR_COMPILER_MODE will not take effect.")
                else:
                    self.vendor_compiler_config = {"mode": vendor_compiler_mode}
