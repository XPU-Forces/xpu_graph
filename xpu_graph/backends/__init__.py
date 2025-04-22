from typing import Optional, Dict, Callable, Any
import torch
from xpu_graph.config import Target
from xpu_graph.utils import logger


def vendor_compiler(
    gm: torch.fx.GraphModule,
    fake_inputs: list,
    target: Target,
    config_dict: Optional[Dict[str, Any]],
    **extra_kwargs
) -> Callable:
    if target == Target.ascend:
        from .ascend import ascend_compile

        logger.info("ascend_compile start...")
        ascend_compiled = ascend_compile(gm, fake_inputs, **extra_kwargs)
        logger.info("ascend_compile complete")
        return ascend_compiled
    elif target == Target.npu:
        from .npu import npu_compile

        logger.info("npu_compile start...")
        npu_compiled = npu_compile(gm, fake_inputs, config_dict, **extra_kwargs)
        logger.info("npu_compile complete")
        return npu_compiled
    elif target == Target.mlu:
        from .mlu import mlu_compile

        logger.info("mlu_compile start...")
        mlu_compiled = mlu_compile(gm, fake_inputs, config_dict, **extra_kwargs)
        logger.info("mlu_compile complete")
        return mlu_compiled

    return gm
