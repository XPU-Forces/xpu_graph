from typing import Dict

import torch
import torch_mlu

from xpu_graph.fx_utils import decompose_for_inductor
from xpu_graph.utils import logger


def mlu_compile(module: torch.nn.Module, example_inputs, **config_dict: Dict) -> torch.nn.Module:
    logger.info("Decompose gm for mlu_inductor")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        module = decompose_for_inductor(module, example_inputs)
    logger.debug(f"After decompose_for_inductor, graph like:\n {module.graph}")

    mode = config_dict.get("mode", "reduce-overhead")
    cpp_wrapper = config_dict.get("cpp_wrapper", False)
    is_inference = config_dict.get("is_inference", False)
    is_backward = config_dict.get("is_backward", False)

    if mode == "cudagraphs":
        from torch._dynamo.backends.cudagraphs import cudagraphs

        return cudagraphs(module, example_inputs)

    from torch import _TorchCompileInductorWrapper
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    options = {}
    dynamic = config_dict.get("dynamic", True)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)

    with torch._inductor.config.patch(inductor_backend.config):
        from packaging import version

        torch_version = version.parse(torch.__version__[:5])
        if cpp_wrapper and torch_version >= version.parse("2.7.0"):
            from torch._inductor.compile_fx import get_cpp_wrapper_config

            config = get_cpp_wrapper_config()
        else:
            config = {}
        with torch._inductor.config.patch(**config):
            compiled_func = compile_fx_inner(
                module, example_inputs, cpp_wrapper=cpp_wrapper, is_inference=is_inference, is_backward=is_backward
            )
    return compiled_func
