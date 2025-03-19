from typing import Dict
import torch
import torch_npu
from .passes.npu.sum_bool import sum_bool_pass

def npu_compile(
    module: torch.nn.Module, inputs, config_dict: Dict
) -> torch.nn.Module:
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor import config
    from torch._inductor import list_mode_options
    if tracing_context := torch._guards.TracingContext.try_get():
        tracing_context.num_mutation_input = module.num_mutation_input
    # default means None
    mode = config_dict.get("mode", "default")
    dynamic = config_dict.get("dynamic", False)
    # torch bug: config values are assumed to be None.
    # Thus we do not use _TorchCompileInductorWrapper to pass custom passes
    # Refer to
    # [1] https://github.com/pytorch/pytorch/issues/139822
    # [2] https://github.com/pytorch/pytorch/pull/139833/files
    # from torch import _TorchCompileInductorWrapper
    # options = {}
    # inductor_backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    current_config = config.shallow_copy_dict()
    mode_dict = list_mode_options(mode, dynamic)
    current_config.update(mode_dict)
    current_config['post_grad_custom_post_pass'] = sum_bool_pass
    compiled_module = compile_fx(
        module, inputs, config_patches=current_config
    )

    return compiled_module

def get_torch_version():
    torch_version = torch.__version__
    if '+' in torch_version:
        return torch_version.split('+')[0]
    return torch_version

if get_torch_version() <= '2.3.1':
    def _patch_inductor_for_aclgraph():
        # Func `num_fw_fixed_arguments` is used to calulate the num of fixed args.
        # Patch it to fix a bug caused by xpu_graph, which will lead to more inplace copy.
        def new_num_fw_fixed_arguments(dynamo_gm_num_inputs: int, aot_fw_gm_num_inputs: int):
            "Computes the number of inputs to the aot fw graph which have fixed addresses (params and buffers)"
            if tracing_context := torch._guards.TracingContext.try_get():
                try:
                    dynamo_gm_num_inputs = tracing_context.num_mutation_input
                except AttributeError:
                    pass

            num_rng_seed_offset_inputs = (
                2 if torch._functorch.config.functionalize_rng_ops else 0
            )
            return aot_fw_gm_num_inputs - dynamo_gm_num_inputs - num_rng_seed_offset_inputs

        torch._inductor.utils.num_fw_fixed_arguments = new_num_fw_fixed_arguments

    _patch_inductor_for_aclgraph()