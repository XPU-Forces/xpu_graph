from typing import Dict

import torch


def npu_compile(module: torch.nn.Module, inputs, config_dict: Dict, **kwargs) -> torch.nn.Module:
    from torch import _TorchCompileInductorWrapper
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    # default means None. In torch, _TorchCompileInductorWrapper's apply_mode just passes.
    mode = config_dict.get("mode", "default")
    options = {}
    dynamic = config_dict.get("dynamic", False)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    with torch._inductor.config.patch(inductor_backend.config):
        compiled_func = compile_fx_inner(module, inputs, **kwargs)
    # DUMMY
    return compiled_func
