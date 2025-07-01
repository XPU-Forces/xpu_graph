from typing import Dict

import torch


def ge_compiler(module: torch.nn.Module, example_inputs, config_dict, **kwargs) -> torch.nn.Module:
    import torch.fx as fx
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)

    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    if config_dict.get("mode", None) == "reduce-overhead":
        config.mode = config_dict["mode"]
        from torch import SymInt

        for ele in example_inputs:
            if isinstance(ele, SymInt):
                raise TypeError("ACL Graph does not support dynamic shape!!")
    else:
        """
        TODO(zhangjihang): We have to use this, cause some case we have to use GE
        """
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.frozen_parameter = True
    npu_backend = tng.get_npu_backend(compiler_config=config)
    compiled_module = npu_backend(module, example_inputs)

    return compiled_module


def inductor_compiler(module: torch.nn.Module, inputs, config_dict: Dict, **kwargs) -> torch.nn.Module:
    from torch import _TorchCompileInductorWrapper
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    # default means None. In torch, _TorchCompileInductorWrapper's apply_mode just passes.
    mode = config_dict.get("mode", "default")
    options = {}
    dynamic = config_dict.get("dynamic", False)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    with torch._inductor.config.patch(inductor_backend.config):
        compiled_func = compile_fx_inner(module, inputs, **kwargs)

    return compiled_func


def npu_compile(module: torch.nn.Module, inputs, config_dict: Dict, **kwargs) -> torch.nn.Module:
    compiler = config_dict.get("compiler", "inductor")
    if compiler == "ge":
        return ge_compiler(module, inputs, config_dict, **kwargs)
    else:
        return inductor_compiler(module, inputs, config_dict, **kwargs)
