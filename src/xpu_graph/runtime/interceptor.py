import copy
import functools
import itertools
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Optional

import torch
from torch.fx import GraphModule
from torch.nn import Module

from xpu_graph.config import get_dump_dir
from xpu_graph.utils import __XPU_GRAPH_ENVS__, logger

from . import XpuGraphRuntimeArtifact, call_func_at_runtime

CASE_CNT = itertools.count()


class InterceptorMode(Enum):
    FAIL_DUMP = "fail_dump"
    FAIL_ERROR = "fail_error"
    FALLBACK = "fallback"
    PASSTHROUGH = "passthrough"


@dataclass
class InterceptorCtx:
    mode: InterceptorMode = InterceptorMode.FAIL_DUMP
    use_golden: bool = True
    check_configs: dict = field(default_factory=dict)

    @staticmethod
    def parse_from(config_str):
        new_ctx = InterceptorCtx()
        if config_str is not None:
            for item in config_str.split(","):
                key, val = item.split("=")
                if key == "mode":
                    new_ctx.mode = InterceptorMode(val)
                elif key in ["rtol", "atol"]:
                    val = float(val)
                    new_ctx.check_configs[key] = val
                elif key in [
                    "allow_subclasses",
                    "equal_nan",
                    "check_device",
                    "check_dtype",
                    "check_layout",
                    "check_stride",
                ]:
                    val = val.lower() in ["true", "1", "on"]
                    new_ctx.check_configs[key] = val
                elif key == "use_golden":
                    new_ctx.use_golden = val.lower() in ["true", "1", "on"]
                else:
                    logger.warning(f"Invalid key: {key}")
        return new_ctx


_CURRENT_INTERCEPT_CTX_ = None


@contextmanager
def reset_intercept_ctx(ctx_str: Optional[str] = None, **kwargs):
    global _CURRENT_INTERCEPT_CTX_
    old_ctx = _CURRENT_INTERCEPT_CTX_
    try:
        new_ctx = replace(InterceptorCtx.parse_from(ctx_str), **kwargs)
        _CURRENT_INTERCEPT_CTX_ = new_ctx
        logger.debug(f"Enable intercept context: {_CURRENT_INTERCEPT_CTX_}")
        yield
    finally:
        _CURRENT_INTERCEPT_CTX_ = old_ctx
        logger.debug(f"Reset intercept context: {_CURRENT_INTERCEPT_CTX_}")


def get_current_intercept_ctx():
    return _CURRENT_INTERCEPT_CTX_


def intercept(target_fn, *, golden_fn, fw_metadata, is_training, mark=0, config_str: Optional[str] = None):
    if config_str is not None and isinstance(config_str, str):
        intercept_ctx = InterceptorCtx.parse_from(config_str)
    else:
        intercept_ctx = InterceptorCtx()

    logger.info(f"Wrapping compiled function with {intercept_ctx}")

    impure = fw_metadata is None or fw_metadata.num_mutated_inp_runtime_indices > 0

    if is_training:
        return AutogradInterceptor(target_fn, golden_fn, mark, impure=impure, intercept_ctx=intercept_ctx)
    else:
        return FunctionInterceptor(target_fn, golden_fn, mark, impure=impure, intercept_ctx=intercept_ctx)


def _invoke_inference(func, inputs):
    outputs = call_func_at_runtime(func, *inputs)
    return outputs


def _invoke_forward(func, inputs):
    with torch.enable_grad():
        outputs = func(*inputs)
    return outputs


def _invoke_backward(inputs, outputs, grad_outputs, inputs_requires_grad):
    inputs_ad_idx = []
    inputs_ad = []
    grad_outputs_ad = []
    outputs_ad = []
    for i, t in enumerate(inputs):
        if inputs_requires_grad[i]:
            inputs_ad_idx.append(i)
            inputs_ad.append(t)
        elif isinstance(t, torch.Tensor) and t.requires_grad:
            grad_outputs_ad.append(grad_outputs[i])
            outputs_ad.append(t)

    for i, t in enumerate(outputs):
        if isinstance(t, torch.Tensor) and t.requires_grad:
            grad_outputs_ad.append(grad_outputs[i + len(inputs)])
            outputs_ad.append(t)

    grad_inputs_ad = torch.autograd.grad(outputs_ad, inputs_ad, grad_outputs_ad, allow_unused=True)

    grad_inputs = [None] * len(inputs)
    for i, g in zip(inputs_ad_idx, grad_inputs_ad):
        grad_inputs[i] = g

    return tuple(grad_inputs)


def compare_tensor_list(base_list, trgt_list, **check_configs):
    diverge_cnt = 0
    for i, (base_t, trgt_t) in enumerate(zip(base_list, trgt_list)):
        try:
            torch.testing.assert_close(trgt_t, base_t, **check_configs)
        except AssertionError as e:
            diverge_cnt += 1
            logger.warning(f"Tensor {i} diverges:\n{e}")
    return diverge_cnt


class FunctionInterceptor(XpuGraphRuntimeArtifact):
    def __init__(self, compiled_fn, golden_fn: Callable, mark=0, impure=True, intercept_ctx=None):
        super().__init__(compiled_fn)
        # Note: The monitor is used for inference.
        # We suppose that original gm has no states (as torch dynamo does, which treats parameters as inputs)
        self.golden_fn = golden_fn
        self.mark = mark
        self.impure = impure
        self.intercept_ctx = intercept_ctx or InterceptorCtx()

    @functools.cached_property
    def runtime_call(self):
        # Similar to what torch._functorch.autograd does, wrap the forward function again
        def monitored_inference(*inputs):
            intercept_ctx = get_current_intercept_ctx() or self.intercept_ctx
            logger.info(f"Monitored inference with {intercept_ctx}")
            if intercept_ctx.mode == InterceptorMode.FALLBACK:
                return _invoke_inference(self.golden_fn, inputs)
            if intercept_ctx.mode == InterceptorMode.PASSTHROUGH:
                return _invoke_inference(self._compiled_func, inputs)
            use_golden = intercept_ctx.use_golden
            ## base func is the actual func in the original autograd graph
            ## trgt func is executed alongside to compare with
            if use_golden:
                base_func, trgt_func = self.golden_fn, self._compiled_func
            else:
                base_func, trgt_func = self._compiled_func, self.golden_fn

            base_inputs = inputs
            if self.impure:
                backup_inputs = copy.deepcopy(base_inputs)
            else:
                backup_inputs = base_inputs

            # restore the random state after golden forward
            if self.impure:
                trgt_inputs = copy.deepcopy(base_inputs)
            else:
                trgt_inputs = base_inputs

            # execute the trgt func first to restore rng state
            with torch.random.fork_rng(device_type=torch._utils._get_available_device_type()):
                trgt_outputs = _invoke_inference(trgt_func, trgt_inputs)

            base_outputs = _invoke_inference(base_func, base_inputs)
            input_diverge_cnt = 0
            if self.impure:
                input_diverge_cnt = compare_tensor_list(base_inputs, trgt_inputs, **intercept_ctx.check_configs)
            output_diverge_cnt = compare_tensor_list(base_outputs, trgt_outputs, **intercept_ctx.check_configs)
            diverge_cnt = input_diverge_cnt + output_diverge_cnt
            if diverge_cnt > 0:
                global CASE_CNT
                case_id = next(CASE_CNT)
                dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_inference_{case_id}")
                logger.warning(
                    f"Summary: The inference pass diverges for {self.golden_fn}\n"
                    f"         input_diverge_cnt: {input_diverge_cnt}, output_diverge_cnt: {output_diverge_cnt}\n"
                    f"         cases saved_to: {dump_path}"
                )
                os.makedirs(dump_path, exist_ok=True)
                if isinstance(self.golden_fn, GraphModule):
                    with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                        mod_str = self.golden_fn.print_readable(
                            print_output=False, include_stride=True, include_device=True
                        )
                        gm_f.write(mod_str)
                if use_golden:
                    golden_inputs, actual_inputs = base_inputs, trgt_inputs
                    golden_outputs, actual_outputs = base_outputs, trgt_outputs
                else:
                    golden_inputs, actual_inputs = trgt_inputs, base_inputs
                    golden_outputs, actual_outputs = trgt_outputs, base_outputs
                dump_glob = {
                    "inputs": backup_inputs,
                    "golden_inputs": golden_inputs,
                    "actual_inputs": actual_inputs,
                    "golden_outputs": golden_outputs,
                    "actual_outputs": actual_outputs,
                }
                torch.save(
                    dump_glob,
                    os.path.join(dump_path, "dump_glob_inf.pt"),
                )
                if intercept_ctx.mode == InterceptorMode.FAIL_ERROR:
                    raise RuntimeError("Inference pass diverges")

            return base_outputs

        if hasattr(self._compiled_func, "zero_grad"):
            monitored_inference.zero_grad = self._compiled_func.zero_grad
        if hasattr(self._compiled_func, "named_parameters"):
            monitored_inference.named_parameters = self._compiled_func.named_parameters
        if hasattr(self._compiled_func, "named_buffers"):
            monitored_inference.named_buffers = self._compiled_func.named_buffers
        return monitored_inference


class AutogradInterceptor(XpuGraphRuntimeArtifact):
    def __init__(self, compiled_fn, golden_fn: Callable, mark=0, impure=True, intercept_ctx=None):
        super().__init__(compiled_fn)
        # Note: The monitor is used for training.
        # We suppose that original gm has no states (as torch dynamo does, which treats parameters as inputs)
        self.golden_fn = golden_fn
        if isinstance(golden_fn, Module):
            assert len(golden_fn.state_dict()) == 0
        self.mark = mark
        self.impure = impure
        self.intercept_ctx = intercept_ctx or InterceptorCtx()

    @functools.cached_property
    def runtime_call(self):
        class MonitoredCompiled(torch.autograd.Function):
            @staticmethod
            def forward(ctx, intercept_ctx, *inputs):
                if intercept_ctx.use_golden:
                    base_func, trgt_func = self.golden_fn, self._compiled_func
                else:
                    base_func, trgt_func = self._compiled_func, self.golden_fn

                inputs_requires_grad = [isinstance(t, torch.Tensor) and t.requires_grad for t in inputs]
                base_inputs = [
                    t.detach().requires_grad_(t.requires_grad) if isinstance(t, torch.Tensor) else t for t in inputs
                ]

                if self.impure:
                    backup_inputs = copy.deepcopy(base_inputs)
                else:
                    backup_inputs = base_inputs

                # restore the random state after golden forward
                if self.impure:
                    trgt_inputs = copy.deepcopy(base_inputs)
                else:
                    trgt_inputs = base_inputs
                with torch.random.fork_rng(device_type=torch._utils._get_available_device_type()):
                    trgt_outputs = _invoke_forward(trgt_func, trgt_inputs)

                base_outputs = _invoke_forward(base_func, base_inputs)
                input_diverge_cnt = 0
                if self.impure:
                    input_diverge_cnt = compare_tensor_list(base_inputs, trgt_inputs, **intercept_ctx.check_configs)
                output_diverge_cnt = compare_tensor_list(base_outputs, trgt_outputs, **intercept_ctx.check_configs)
                diverge_cnt = input_diverge_cnt + output_diverge_cnt
                if diverge_cnt > 0:
                    global CASE_CNT
                    case_id = next(CASE_CNT)
                    dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_forward_{case_id}")
                    logger.warning(
                        f"Summary: The forward pass diverges for {self.golden_fn}\n"
                        f"         input_diverge_cnt: {input_diverge_cnt}, output_diverge_cnt: {output_diverge_cnt}\n"
                        f"         cases saved_to: {dump_path}"
                    )
                    os.makedirs(dump_path, exist_ok=True)
                    if isinstance(self.golden_fn, GraphModule):
                        with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                            mod_str = self.golden_fn.print_readable(
                                print_output=False, include_stride=True, include_device=True
                            )
                            gm_f.write(mod_str)

                    if intercept_ctx.use_golden:
                        golden_inputs, actual_inputs = base_inputs, trgt_inputs
                        golden_outputs, actual_outputs = base_outputs, trgt_outputs
                    else:
                        golden_inputs, actual_inputs = trgt_inputs, base_inputs
                        golden_outputs, actual_outputs = trgt_outputs, base_outputs

                    dump_glob = {
                        "inputs": backup_inputs,
                        "golden_inputs": golden_inputs,
                        "actual_inputs": actual_inputs,
                        "golden_outputs": golden_outputs,
                        "actual_outputs": actual_outputs,
                    }
                    torch.save(
                        dump_glob,
                        os.path.join(dump_path, "dump_glob_fwd.pt"),
                    )
                    if intercept_ctx.mode == InterceptorMode.FAIL_ERROR:
                        raise RuntimeError("Forward pass diverges")

                ctx.saved_states = {
                    "base_inputs": base_inputs,
                    "trgt_inputs": trgt_inputs,
                    "base_outputs": base_outputs,
                    "trgt_outputs": trgt_outputs,
                    "inputs_requires_grad": inputs_requires_grad,
                }
                ctx.intercept_ctx = intercept_ctx
                inputs_after = [
                    (
                        t_after.detach().requires_grad_(True)
                        if isinstance(t_after, torch.Tensor) and t_after.requires_grad and not req_grad
                        else None
                    )
                    for t_after, req_grad in zip(base_inputs, inputs_requires_grad)
                ]
                outputs = [
                    t.detach().requires_grad_(True) if isinstance(t, torch.Tensor) and t.requires_grad else t
                    for t in base_outputs
                ]
                # Note: wrap the results with a tuple, thus the backward function is guaranteed to be triggered
                return tuple(inputs_after + outputs)

            @staticmethod
            def backward(ctx, *grad_outputs):
                intercept_ctx = ctx.intercept_ctx
                logger.info(f"Monitored backward with {intercept_ctx}")

                base_inputs = ctx.saved_states["base_inputs"]
                trgt_inputs = ctx.saved_states["trgt_inputs"]
                base_outputs = ctx.saved_states["base_outputs"]
                trgt_outputs = ctx.saved_states["trgt_outputs"]
                inputs_requires_grad = ctx.saved_states["inputs_requires_grad"]

                trgt_grad_inputs = _invoke_backward(trgt_inputs, trgt_outputs, grad_outputs, inputs_requires_grad)
                base_grad_inputs = _invoke_backward(base_inputs, base_outputs, grad_outputs, inputs_requires_grad)
                grad_diverge_cnt = compare_tensor_list(
                    base_grad_inputs, trgt_grad_inputs, **intercept_ctx.check_configs
                )
                if grad_diverge_cnt > 0:
                    global CASE_CNT
                    case_id = next(CASE_CNT)
                    dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_backward_{case_id}")
                    logger.warning(
                        f"Summary: The backward pass diverges for {self.golden_fn}\n"
                        f"         grad_diverge_cnt: {grad_diverge_cnt}\n"
                        f"         cases saved_to: {dump_path}"
                    )
                    os.makedirs(dump_path, exist_ok=True)
                    if isinstance(self.golden_fn, GraphModule):
                        with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                            mod_str = self.golden_fn.print_readable(
                                print_output=False, include_stride=True, include_device=True
                            )
                            gm_f.write(mod_str)
                    if intercept_ctx.use_golden:
                        golden_inputs, actual_inputs = base_inputs, trgt_inputs
                        golden_grad_inputs, actual_grad_inputs = base_grad_inputs, trgt_grad_inputs
                    else:
                        golden_inputs, actual_inputs = trgt_inputs, base_inputs
                        golden_grad_inputs, actual_grad_inputs = trgt_grad_inputs, base_grad_inputs
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
                    if intercept_ctx.mode == InterceptorMode.FAIL_ERROR:
                        raise RuntimeError("Backward pass diverges")
                return None, *base_grad_inputs

        # Similar to what torch._functorch.autograd does, wrap the forward function again
        def monitored_forward(*inputs):
            intercept_ctx = get_current_intercept_ctx() or self.intercept_ctx
            logger.info(f"Monitored forward with {intercept_ctx}")
            if intercept_ctx.mode == InterceptorMode.FALLBACK:
                return self.golden_fn(*inputs)
            if intercept_ctx.mode == InterceptorMode.PASSTHROUGH:
                return self._compiled_func(*inputs)

            outputs = MonitoredCompiled.apply(intercept_ctx, *inputs)
            after_inputs = outputs[: len(inputs)]
            for t, t_after in zip(inputs, after_inputs):
                # filter out inplace mutations with grad
                if t_after is not None:
                    t.copy_(t_after)

            outputs = outputs[len(inputs) :]
            return outputs

        if hasattr(self._compiled_func, "zero_grad"):
            monitored_forward.zero_grad = self._compiled_func.zero_grad
        if hasattr(self._compiled_func, "named_parameters"):
            monitored_forward.named_parameters = self._compiled_func.named_parameters
        if hasattr(self._compiled_func, "named_buffers"):
            monitored_forward.named_buffers = self._compiled_func.named_buffers
        return monitored_forward


from torch.utils._python_dispatch import TorchDispatchMode


class OpInterceptor(TorchDispatchMode):
    def __init__(self, golden_funcs, mark=None, dispatch_key=None, **check_configs):
        super().__init__(dispatch_key)
        self.golden_funcs = golden_funcs
        self.mark = mark if mark is not None else "op"
        self.check_configs = check_configs

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if func in self.golden_funcs:
            golden_fn = self.golden_funcs[func]
            logger.info(f"Monitored op: {func}")

            actual_inputs = args
            golden_inputs = copy.deepcopy(actual_inputs)
            with torch.random.fork_rng(device_type=torch._utils._get_available_device_type()):
                golden_outputs = golden_fn(*golden_inputs, **(kwargs or {}))
            actual_outputs = func(*args, **(kwargs or {}))
            input_diverge_cnt = compare_tensor_list(actual_inputs, golden_inputs, **self.check_configs)
            output_diverge_cnt = compare_tensor_list(actual_outputs, golden_outputs, **self.check_configs)
            if input_diverge_cnt > 0 or output_diverge_cnt > 0:
                global CASE_CNT
                case_id = next(CASE_CNT)
                dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_op_{case_id}")
                logger.warning(
                    f"Summary: The op diverges for {func}\n"
                    f"         input_diverge_cnt: {input_diverge_cnt}, output_diverge_cnt: {output_diverge_cnt}\n"
                    f"         cases saved_to: {dump_path}"
                )
                os.makedirs(dump_path, exist_ok=True)
                dump_glob = {
                    "golden_inputs": golden_inputs,
                    "actual_inputs": actual_inputs,
                    "golden_outputs": golden_outputs,
                    "actual_outputs": actual_outputs,
                }
                torch.save(
                    dump_glob,
                    os.path.join(dump_path, "dump_glob_op.pt"),
                )
            return actual_outputs
        return func(*args, **(kwargs or {}))
