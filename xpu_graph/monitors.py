import copy
import itertools
import os
from typing import Callable

import torch
from torch.fx import GraphModule
from torch.nn import Module

from xpu_graph.config import get_dump_dir
from xpu_graph.utils import logger

CASE_CNT = itertools.count()


def _invoke_forward(func, inputs):
    with torch.enable_grad():
        outputs = func(*inputs)
    return outputs


def _invoke_backward(func, inputs, outputs, grad_outputs):
    inputs_ad_idx = [i for i, t in enumerate(inputs) if isinstance(t, torch.Tensor) and t.requires_grad]
    inputs_ad = [t for t in inputs if isinstance(t, torch.Tensor) and t.requires_grad]
    outputs_ad = [t for t in outputs if isinstance(t, torch.Tensor) and t.requires_grad]

    outputs_ad_idx = [i for i, t in enumerate(outputs) if isinstance(t, torch.Tensor) and t.requires_grad]
    grad_outputs_ad = [grad_outputs[idx] for idx in outputs_ad_idx]
    grad_inputs_ad = torch.autograd.grad(outputs_ad, inputs_ad, grad_outputs_ad, allow_unused=True)

    grad_inputs = [None] * len(inputs)
    for i, g in zip(inputs_ad_idx, grad_inputs_ad):
        grad_inputs[i] = g

    return tuple(grad_inputs)


class AutogradMonitor:
    def __init__(self, golden_fn: Callable, mark=0):
        # Note: The monitor is used for training.
        # We suppose that original gm has no states (as torch dynamo does, which treats parameters as inputs)
        self.golden_fn = golden_fn
        if isinstance(golden_fn, Module):
            assert len(golden_fn.state_dict()) == 0
        self.mark = mark

    def guard(self, compiled_fn):
        class MonitoredCompiled(torch.autograd.Function):
            golden_func = self.golden_fn
            target_func = compiled_fn

            @staticmethod
            def forward(ctx, *inputs):
                logger.debug("Monitored forward")
                actual_inputs = [
                    t.detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t for t in inputs
                ]
                # Currently, we assume no mutations on inputs

                # restore the random state after golden forward
                golden_inputs = copy.deepcopy(actual_inputs)
                with torch.random.fork_rng(device_type=torch._utils._get_available_device_type()):
                    golden_outputs = _invoke_forward(MonitoredCompiled.golden_func, golden_inputs)

                actual_outputs = _invoke_forward(MonitoredCompiled.target_func, actual_inputs)
                try:
                    torch.testing.assert_close(actual_inputs, golden_inputs)
                    torch.testing.assert_close(actual_outputs, golden_outputs)
                except AssertionError as e:
                    global CASE_CNT
                    case_id = next(CASE_CNT)
                    dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_forward_{case_id}")
                    logger.warning(
                        f"The forward pass diverges for {MonitoredCompiled.golden_func}\ncases saved_to: {dump_path}\nError: {e}"
                    )
                    os.makedirs(dump_path, exist_ok=True)
                    if isinstance(MonitoredCompiled.golden_func, GraphModule):
                        with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                            mod_str = MonitoredCompiled.golden_func.print_readable(
                                print_output=False, include_stride=True, include_device=True
                            )
                            gm_f.write(mod_str)

                    dump_glob = {
                        "golden_inputs": golden_inputs,
                        "actual_inputs": actual_inputs,
                        "golden_outputs": golden_outputs,
                        "actual_outputs": actual_outputs,
                    }
                    torch.save(
                        dump_glob,
                        os.path.join(dump_path, "dump_glob_fwd.pt"),
                    )

                ctx.saved_states = {
                    "golden_inputs": golden_inputs,
                    "actual_inputs": actual_inputs,
                    "golden_outputs": golden_outputs,
                    "actual_outputs": actual_outputs,
                }

                # Note: wrap the results with a tuple, thus the backward function is guaranteed to be triggered
                return tuple(
                    t.detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t
                    for t in actual_outputs
                )

            @staticmethod
            def backward(ctx, *grad_outputs):
                logger.debug("Monitored backward")

                golden_inputs = ctx.saved_states["golden_inputs"]
                actual_inputs = ctx.saved_states["actual_inputs"]
                golden_outputs = ctx.saved_states["golden_outputs"]
                actual_outputs = ctx.saved_states["actual_outputs"]

                golden_grad_inputs = _invoke_backward(
                    MonitoredCompiled.golden_func, golden_inputs, golden_outputs, grad_outputs
                )
                actual_grad_inputs = _invoke_backward(
                    MonitoredCompiled.target_func, actual_inputs, actual_outputs, grad_outputs
                )
                try:
                    torch.testing.assert_close(actual_grad_inputs, golden_grad_inputs)  # , atol=0, rtol=0)
                except AssertionError as e:
                    global CASE_CNT
                    case_id = next(CASE_CNT)
                    dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_backward_{case_id}")
                    logger.warning(
                        f"The backward pass diverges for {MonitoredCompiled.golden_func}\ncases saved_to: {dump_path}\nError: {e}"
                    )
                    os.makedirs(dump_path, exist_ok=True)
                    if isinstance(MonitoredCompiled.golden_func, GraphModule):
                        with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                            mod_str = MonitoredCompiled.golden_func.print_readable(
                                print_output=False, include_stride=True, include_device=True
                            )
                            gm_f.write(mod_str)
                    dump_glob = {
                        "golden_inputs": golden_inputs,
                        "actual_inputs": actual_inputs,
                        "grad_outputs": grad_outputs,
                        "golden_grad_inputs": golden_grad_inputs,
                        "actual_grad_inputs": actual_grad_inputs,
                    }
                    torch.save(
                        dump_glob,
                        os.path.join(dump_path, "dump_glob_bwd.pt"),
                    )
                return actual_grad_inputs

        # Similar to what torch._functorch.autograd does, wrap the forward function again
        def monitored_forward(*inputs):
            return MonitoredCompiled.apply(*inputs)

        if hasattr(compiled_fn, "zero_grad"):
            monitored_forward.zero_grad = compiled_fn.zero_grad
        if hasattr(compiled_fn, "named_parameters"):
            monitored_forward.named_parameters = compiled_fn.named_parameters
        if hasattr(compiled_fn, "named_buffers"):
            monitored_forward.named_buffers = compiled_fn.named_buffers
        return monitored_forward
