import functools
import os

from xpu_graph.utils import logger


def wrap_artifact_at_runtime(
    config,
    compiled_func,
    dynamo_gm,
    fw_metadata=None,
):
    if config.enable_interceptor is not None:
        from xpu_graph.runtime.interceptor import intercept

        logger.info("Wrapping compiled function with interceptor")
        return intercept(
            compiled_func,
            golden_fn=dynamo_gm,
            fw_metadata=fw_metadata,
            is_training=config.is_training,
            mark=os.environ.get("RANK", "0"),
            config_str=config.enable_interceptor,
        )

    return XpuGraphRuntimeArtifact(compiled_func)


def call_func_at_runtime(func, *args):
    if getattr(func, "_boxed_call", False):
        # Further reading: what is boxed_call? See discussion in : https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
        return func(list(args))
    return func(*args)


class XpuGraphRuntimeArtifact:
    def __init__(self, compiled_func):
        self._compiled_func = compiled_func

    def __call__(self, *args):
        return self.runtime_call(*args)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "runtime_call" in state:
            del state["runtime_call"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @functools.cached_property
    def runtime_call(self):
        return functools.partial(call_func_at_runtime, self._compiled_func)
