from typing import Dict

import torch
import torch_mlu


def mlu_compile(module: torch.nn.Module, example_inputs, config_dict: Dict, **kwargs) -> torch.nn.Module:
    mode = config_dict.get("mode", "reduce-overhead")
    cpp_wrapper = config_dict.get("cpp_wrapper", False)

    if mode == "cudagraphs":
        from torch._dynamo.backends.cudagraphs import cudagraphs

        return cudagraphs(module, example_inputs)

    from torch import _TorchCompileInductorWrapper
    from torch._inductor.compile_fx import (
        compile_fx,
        compile_fx_inner,
        get_cpp_wrapper_config,
    )

    options = {}
    dynamic = config_dict.get("dynamic", True)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    with torch._inductor.config.patch(inductor_backend.config):
        from packaging import version

        torch_version = version.parse(torch.__version__[:5])
        config = get_cpp_wrapper_config() if cpp_wrapper and torch_version >= version.parse("2.7.0") else {}
        with torch._inductor.config.patch(**config):
            compiled_func = compile_fx_inner(module, example_inputs, cpp_wrapper=cpp_wrapper, **kwargs)
    return compiled_func
