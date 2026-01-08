import operator
from typing import Dict

import torch
import torch.fx
from torch.fx.node import Node
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree

from xpu_graph.utils import logger


class DeviceGraphOperatorSupport(OperatorSupport):
    def __init__(self, target_device: str):
        super().__init__()
        self.target_device = target_device

    def is_node_supported(self, submodules, node: Node) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return False

        if node.target is operator.getitem:
            return True

        if node.op == "call_function" or node.op == "call_method" or node.op == "call_module":
            namespace = getattr(node.target, "namespace", "dummy")
            if namespace in ["torch_npu_triton", "torch_mlu_triton"]:
                return True
            # Note(Zijun): Not sure about how to decide whether a Node is xpu-ops
            if hasattr(node.target, "__module__") and "xpu_ops" in node.target.__module__:
                return True

        found_invalid_device = False

        def get_val(meta):
            return meta.get("val", meta.get("fake_result", None))

        def check_device(t):
            nonlocal found_invalid_device
            if found_invalid_device:
                return
            if isinstance(t, torch.Tensor):
                if t.device.type != self.target_device:
                    found_invalid_device = True

        for input_node in node.all_input_nodes:
            _pytree.tree_map_(check_device, get_val(input_node.meta))
            if found_invalid_device:
                return False

        _pytree.tree_map_(check_device, get_val(node.meta))

        return not found_invalid_device


def device_graph_compiler(module: torch.nn.Module, example_inputs, target, **config_dict: Dict) -> torch.nn.Module:
    # step1: FakeTensor prop
    # Note(Zijun): not sure whether it's necessary, due to pass_manager has done it before
    # fake_propagator = FakeTensorProp(module)
    # fake_propagator.propagate(*example_inputs)

    # step2: Partitioning
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

    supported_ops = DeviceGraphOperatorSupport(target_device=target)
    partitioner = CapabilityBasedPartitioner(module, supported_ops, allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    fused_module = partitioner.fuse_partitions(partitions)

    from xpu_graph.config import Target
    from xpu_graph.device_graph_runner import GraphRunner

    BackendRunnerClass = GraphRunner[Target(target)]

    # Using Lazy mode here because potential in-place operations in the graph might alter the global state
    class _LazyXPUGraph(torch.nn.Module):
        def __init__(self, target_module: torch.nn.Module, compipler_config: Dict):
            super().__init__()
            self.runner = BackendRunnerClass(target_module, None, None)
            self.config = compipler_config
            self._is_initialized = False
            self.copy_tensor_pos = []

        def _setup_callbacks(self, args, kwarg):
            flat_args, _ = _pytree.tree_flatten((args, kwarg))
            self.copy_tensor_pos = []
            for i, v in enumerate(flat_args):
                if isinstance(v, torch.Tensor):
                    self.copy_tensor_pos.append(i)

            def init_param_callback(*args, **kwargs):
                flat, _ = _pytree.tree_flatten((args, kwargs))
                return [flat[i] for i in self.copy_tensor_pos]

            def copy_param_callback(input_buffers, *args, **kwargs):
                flat, _ = _pytree.tree_flatten((args, kwargs))
                runtime_tensors = [flat[i] for i in self.copy_tensor_pos]

                if len(runtime_tensors) != len(input_buffers):
                    return False
                try:
                    for buf, rt in zip(input_buffers, runtime_tensors):
                        if not isinstance(rt, torch.Tensor):
                            return False
                        buf.copy_(rt, non_blocking=True)
                    return True
                except Exception:
                    return False

            self.runner._init_param_callback = init_param_callback
            self.runner._copy_param_callback = copy_param_callback
            self._is_initialized = True

        def forward(self, *args, **kwargs):
            if not self._is_initialized:
                self._setup_callbacks(args, kwargs)
                memory_pool = self.config.get("memory_pool", None)
                clone_args = bool(self.config.get("clone_args", True))
                self.runner.capture(*args, memory_pool=memory_pool, clone_args=clone_args, **kwargs)

                # return self.runner._output
                # FIX: Force a replay to ensure correct output values
                return self.runner.forward(*args, **kwargs)

            return self.runner.forward(*args, **kwargs)

    # step4: Replace fused_module's submodules with LazyXPUGraph
    for name, submodule in fused_module.named_children():
        if isinstance(submodule, torch.fx.GraphModule):
            lazy_runner = _LazyXPUGraph(submodule, config_dict)
            setattr(fused_module, name, lazy_runner)

    return fused_module
