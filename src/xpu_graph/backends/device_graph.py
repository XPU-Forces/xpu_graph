import pickle
from typing import Dict

import torch

def device_graph_compiler(module: torch.nn.Module, example_inputs, target, **config_dict: Dict) -> torch.nn.Module:
    
    if example_inputs is None:
        raise ValueError("device_graph_compiler requires example_inputs for capture().")
    if not isinstance(example_inputs, (tuple, list)):
        example_args = (example_inputs,)
    else:
        example_args = tuple(example_inputs)

    import torch.utils._pytree as pytree
    
    ex_flat, _ = pytree.tree_flatten((example_args, {}))
    copy_tensor_pos = []
    for i, v in enumerate(ex_flat):
        if isinstance(v, torch.nn.Parameter):
            continue
        if isinstance(v, torch.Tensor):
            copy_tensor_pos.append(i)

    from xpu_graph.config import Target
    from xpu_graph.device_graph_runner import GraphRunner
    
    clone_args = bool(config_dict.get("clone_args", True))
    memory_pool = config_dict.get("memory_pool", None)

    # NOTE: We wrap with a lazy module because torch.compile invokes backends with FakeTensors;
    # device graph capture must run with real device tensors at runtime
    class _LazyNPUGraph(torch.nn.Module):
        def __init__(self, callable_target: torch.nn.Module, copy_tensor_pos):
            super().__init__()
            self.copy_tensor_pos = copy_tensor_pos
            if target == "npu":
                self._runner = GraphRunner[Target.npu](
                    callable_target,
                    init_param_callback=self._init_param_callback,
                    copy_param_callback=self._copy_param_callback,
                )
            elif target == "mlu":
                self._runner = GraphRunner[Target.mlu](
                    callable_target,
                    init_param_callback=self._init_param_callback,
                    copy_param_callback=self._copy_param_callback,
                )
            else:
                raise ValueError("Device Graph Runner target not define.")
            
            self._captured = False
            self._warmed_up = False

        @staticmethod
        def _init_param_callback(*args, **kwargs):
            flat, _ = pytree.tree_flatten((args, kwargs))
            # log for debug
            print("init_param_callback: flat lens", len(flat), "copy_tensor_pos", copy_tensor_pos)
            # Return the exact Tensor objects used during capture (after clone_args mapping inside capture()).
            return [flat[i] for i in copy_tensor_pos]

        @staticmethod
        def _copy_param_callback(input_buffers, *args, **kwargs):
            flat, _ = pytree.tree_flatten((args, kwargs))
            runtime_tensors = [flat[i] for i in copy_tensor_pos]
            # log for debug
            print("copy_param_callback: buffer shapes", [b.shape for b in input_buffers], "runtime shapes", [r.shape for r in runtime_tensors])
            if len(runtime_tensors) != len(input_buffers):
                return False
            try:
                for buf, rt in zip(input_buffers, runtime_tensors):
                    if not isinstance(rt, torch.Tensor):
                        return False
                    # buf should be a non-Parameter tensor buffer, so in-place copy is OK.
                    buf.copy_(rt)
                # torch.npu.synchronize()
            except Exception:
                return False
            return True

        def forward(self, *args, **kwargs):
            if not self._warmed_up:
                self._warmed_up = True
                return self._runner._callable_target(*args, **kwargs)

            if not self._captured:
                self._runner.capture(*args, clone_args=clone_args, memory_pool=memory_pool, **kwargs)
                self._captured = True
            
            return self._runner(*args, **kwargs)

    return _LazyNPUGraph(module, copy_tensor_pos)
    