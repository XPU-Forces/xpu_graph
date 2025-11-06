import importlib
from typing import Any, Callable, Dict, Optional

import torch

from xpu_graph.config import Target
from xpu_graph.utils import logger


def is_empty_vendor_cfg(vendor_cfg):
    __KEY_TO_IGNORE__ = {"is_inference", "is_backward"}
    if set(vendor_cfg.keys()) == __KEY_TO_IGNORE__:
        return True
    else:
        return False


def vendor_compiler(
    gm: torch.fx.GraphModule, fake_inputs: list, target: Target, **config_dict: Dict[str, Any]
) -> Callable:
    try:
        target_mod = importlib.import_module(f".{target.value}", __package__)
    except Exception:
        logger.warning(f"{target.value}_compiler not found, return gm")
        return gm

    compile_fn = getattr(target_mod, f"{target.value}_compile")
    if (
        is_empty_vendor_cfg(config_dict)
        and (default_dict := getattr(target_mod, "__DEFAULT_VENDOR_CONFIG__", None)) is not None
    ):
        config_dict = default_dict
    logger.info(f"{target.value}_compile start...")
    xpu_compiled = compile_fn(gm, fake_inputs, **config_dict)
    logger.info(f"{target.value}_compile complete")
    return xpu_compiled
