from typing import Dict

import torch

from xpu_graph.utils import logger, recursive_set_obj


def ge_compiler(module: torch.nn.Module, example_inputs, **config_dict: Dict) -> torch.nn.Module:
    import torch.fx as fx
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)

    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    recursive_set_obj(config_dict, config)
    if config_dict.get("mode", None) == "reduce-overhead":
        config.mode = config_dict["mode"]
        from torch import SymInt

        for ele in example_inputs:
            if isinstance(ele, SymInt):
                raise TypeError("ACL Graph does not support dynamic shape!!")

        if config_dict.get("use_custom_pool", None):
            config.aclgraph_config.use_custom_pool = config_dict["use_custom_pool"]
    else:
        """
        TODO(zhangjihang): We have to use this, cause some case we have to use GE
        """
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.frozen_parameter = True

    npu_backend = tng.get_npu_backend(compiler_config=config)
    compiled_module = npu_backend(module, example_inputs)

    return compiled_module


def inductor_compiler(module: torch.nn.Module, inputs, **config_dict: Dict) -> torch.nn.Module:
    logger.info("Decompose gm for npu_inductor")
    from xpu_graph.fx_utils import decompose_for_inductor

    module = decompose_for_inductor(module, inputs)
    logger.debug(f"After decompose_for_inductor, graph like:\n {module.graph}")

    from torch import _TorchCompileInductorWrapper
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner

    is_inference = config_dict.get("is_inference", False)
    is_backward = config_dict.get("is_backward", False)

    # default means None. In torch, _TorchCompileInductorWrapper's apply_mode just passes.
    mode = config_dict.get("mode", "default")
    options = {}
    dynamic = config_dict.get("dynamic", False)
    inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)

    with torch._inductor.config.patch(inductor_backend.config):
        compiled_func = compile_fx_inner(module, inputs, is_inference=is_inference, is_backward=is_backward)

    return compiled_func


def npu_compile(module: torch.nn.Module, inputs, **config_dict: Dict) -> torch.nn.Module:
    compiler = config_dict.get("compiler", "inductor")
    if compiler == "ge":
        return ge_compiler(module, inputs, **config_dict)
    else:
        return inductor_compiler(module, inputs, **config_dict)
