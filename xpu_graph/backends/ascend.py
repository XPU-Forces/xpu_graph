import torch
import torch.fx as fx

def npu_compile(module: torch.nn.Module, example_inputs) -> torch.nn.Module:
    torch.npu.set_compile_mode(jit_compile=False)

    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    from torchair.configs.compiler_config import CompilerConfig

    """
    TODO(zhangjihang): We have to use this, cause some case we have to use GE
    """
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    config.experimental_config.frozen_parameter = True

    """Add this config to enable AclGraph in npu backend."""
    config.mode = "reduce-overhead"

    npu_backend = tng.get_npu_backend(compiler_config=config)
    compiled_module = npu_backend(module, example_inputs)

    return compiled_module
