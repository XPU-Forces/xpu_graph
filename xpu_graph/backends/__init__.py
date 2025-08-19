import importlib
from typing import Any, Callable, Dict, Optional

import torch

from xpu_graph.config import Target
from xpu_graph.utils import logger


def vendor_compiler(
    gm: torch.fx.GraphModule, fake_inputs: list, target: Target, **config_dict: Dict[str, Any]
) -> Callable:
    try:
        target_mod = importlib.import_module(f".{target.value}", __package__)
        compile_fn = getattr(target_mod, f"{target.value}_compile")
        logger.info(f"{target.value}_compile start...")
        xpu_compiled = compile_fn(gm, fake_inputs, **config_dict)
        logger.info(f"{target.value}_compile complete")
        return xpu_compiled
    except Exception:
        logger.warning(f"{target.value}_compiler not found, return gm")
        return gm
