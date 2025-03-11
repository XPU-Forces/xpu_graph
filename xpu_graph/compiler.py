from typing import Callable, overload

import torch

from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

from .passes.pass_manager import PassManager
from .passes.patterns.pattern import Pattern
from .config import XpuGraphConfig, Target, OptLevel
from .utils import logger, setup_logger
from .cache import XpuGraphCache, default_cache
import logging


def optimize_graph(gm, sample_inputs, config=None):
    # Create default config if none provided
    if config is None:
        config = XpuGraphConfig(
            target=Target.mlu, enable_cache=False, opt_level=OptLevel.level2
        )

    # Setup logging based on config
    setup_logger(logging.DEBUG if config.debug else logging.INFO)

    # Create fake inputs for optimization
    fake_mode = FakeTensorMode()
    fake_mode.allow_non_fake_inputs = True
    fake_inputs = [
        fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
        for x in sample_inputs
    ]

    with fake_mode:
        logger.debug(f"before xpu_optimize_graph, graph like:\n {gm.graph}")
        logger.info(f"before xpu_optimize_graph, nodes num: {len(gm.graph.nodes)}")

        pass_manager = PassManager(config)
        xpu_optimized = pass_manager(gm, fake_inputs)

        logger.debug(f"after xpu_optimize_graph, graph like:\n {xpu_optimized.graph}")
        logger.info(
            f"after xpu_optimize_graph, nodes num: {len(xpu_optimized.graph.nodes)}"
        )

    return xpu_optimized


class XpuGraph:
    def __init__(
        self,
        config: XpuGraphConfig = XpuGraphConfig(),
        cache: XpuGraphCache = None,
    ):
        self._config = config
        # Setup logging based on config
        setup_logger(logging.DEBUG if self._config.debug else logging.INFO)

        if self._config.freeze:
            # The configuration in this inductor affects the return value of is_parameter_freezing(),
            # thereby influencing the process of generating the fx_graph in dynamo. The current code
            # in the community is not very clean, and it would be more reasonable to place this
            # configuration under dynamo. You can refer to this link for more information.
            # https://github.com/pytorch/pytorch/blob/release/2.5/torch/_dynamo/utils.py#L3061
            torch._inductor.config.freezing = True

        self._pass_manager = PassManager(self._config)
        self._cache = (
            cache
            if cache and config.enable_cache
            else default_cache()
            if config.enable_cache
            else None
        )

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        def _compiler(gm, sample_inputs):
            if self._config.skip_all_pass:
                return gm

            # Create fake inputs for optimization
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(sample_inputs)
            fake_mode.allow_non_fake_inputs = True
            fake_inputs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in sample_inputs
            ]

            with fake_mode:
                logger.debug(f"before xpu_graph, graph like:\n {gm.graph}")
                logger.info(f"before xpu_graph, nodes num: {len(gm.graph.nodes)}")
                logger.info("xpu_graph passes start...")

                # Try to load from cache or run pass manager
                if self._config.enable_cache and self._cache:
                    hashkey = self._cache.cache_key(gm, fake_inputs, self._config)
                    xpu_compiled = self._cache.load_gm(hashkey)
                    if xpu_compiled is None:
                        xpu_compiled = self._pass_manager(gm, fake_inputs)
                        xpu_compiled = self._cache.save_gm(hashkey, xpu_compiled)
                else:
                    xpu_compiled = self._pass_manager(gm, fake_inputs)

                logger.debug(f"after xpu_graph, graph like:\n {xpu_compiled.graph}")
                logger.info("xpu_graph passes complete")
                logger.info(
                    f"after xpu_graph, nodes num: {len(xpu_compiled.graph.nodes)}"
                )

                # Apply vendor compiler if configured
                if self._config.vendor_compiler_config:
                    from .backends import vendor_compiler

                    return vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        self._config.target,
                        self._config.vendor_compiler_config,
                    )

            return xpu_compiled

        if self._config.freeze:
            logger.info("unlift graph start...")
            lifted_gm, gs = aot_export_module(
                dynamo_gm, example_inputs, trace_joint=False
            )

            logger.debug(f"before unlift, graph like:\n {lifted_gm.graph}")

            from xpu_graph.fx_utils import unlift_gm

            unlifted_gm = unlift_gm(dynamo_gm, lifted_gm, gs)
            logger.info("unlift graph complete")
            logger.debug(f"after unlift, graph like:\n {unlifted_gm.graph}")

            return _compiler(unlifted_gm, example_inputs)
        else:
            return aot_autograd(fw_compiler=_compiler)(dynamo_gm, example_inputs)

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()
