from typing import Callable, overload

import torch

from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_export_module
from torch._subclasses.fake_tensor import FakeTensorMode

from .passes.pass_manager import PassManager
from .passes.patterns.pattern import Pattern
from .config import XpuGraphConfig, Target, OptLevel
from .utils import logger, setup_logger, local_logger, get_nodes_statistics
from .cache import XpuGraphCache, default_cache
from .fx_utils import FxStage, unlift_exported_gm
import logging
from functools import partial


def optimize_graph(gm, sample_inputs, config=None):
    # Create default config if none provided
    if config is None:
        config = XpuGraphConfig(
            is_training=False,
            target=Target.none,
            enable_cache=False,
            opt_level=OptLevel.level2,
        )
    config._reset_config_with_env()

    # Setup logging based on config
    setup_logger(logging.DEBUG if config.debug else logging.INFO)

    logger.info(f"{config}")

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
        xpu_optimized = pass_manager(gm, fake_inputs, stage=FxStage.inference)

        logger.debug(f"after xpu_optimize_graph, graph like:\n {xpu_optimized.graph}")
        logger.info(
            f"after xpu_optimize_graph, nodes num: {len(xpu_optimized.graph.nodes)}"
        )

    return xpu_optimized


class XpuGraph:
    def __init__(
        self,
        config: XpuGraphConfig,
        cache: XpuGraphCache = None,
    ):
        config._reset_config_with_env()
        self._config = config
        # Setup logging based on config
        setup_logger(logging.DEBUG if self._config.debug else logging.INFO)

        logger.info(f"{config}")

        self._pass_manager = PassManager(self._config)
        self._cache = (
            cache
            if cache and config.enable_cache
            else default_cache() if config.enable_cache else None
        )

        self._set_context()

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        def _compiler(gm, sample_inputs, stage: FxStage):

            # Create fake inputs for optimization
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(sample_inputs)
            fake_mode.allow_non_fake_inputs = True
            fake_inputs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in sample_inputs
            ]

            with fake_mode:
                with local_logger("before"):
                    logger.info(
                        f"before xpu_graph, nodes statistics: {get_nodes_statistics(gm)}"
                    )
                    logger.debug(f"before xpu_graph, graph like:\n {gm.graph}")
                    logger.info(f"xpu_graph passes start {stage}...")

                if stage == FxStage.pregrad:
                    logger.debug(f"before decompose: graph like:\n {gm.graph}")
                    logger.info("decompose graph start...")
                    from torch.fx.experimental.proxy_tensor import make_fx

                    gm = make_fx(
                        gm,
                        tracing_mode="fake",
                        pre_dispatch=True,
                        record_module_stack=True,
                    )(*fake_inputs)
                    logger.info("decompose graph complete")
                    logger.debug(f"after decompose, graph like:\n {gm.graph}")

                if self._config.enable_cache:
                    hashkey = self._cache.cache_key(
                        gm, fake_inputs, self._config, stage
                    )
                    xpu_compiled = self._cache.load_gm(hashkey)
                    if xpu_compiled is None:
                        xpu_compiled = self._pass_manager(gm, fake_inputs, stage)
                        if self._config.target != Target("ascend"):
                            xpu_compiled = self._cache.save_gm(hashkey, xpu_compiled)
                else:
                    xpu_compiled = self._pass_manager(gm, fake_inputs, stage)

                with local_logger("after"):
                    logger.info("xpu_graph passes complete")
                    logger.info(
                        f"after xpu_graph, nodes statistics: \n{get_nodes_statistics(xpu_compiled)}"
                    )
                    logger.debug(f"after xpu_graph, graph like:\n {xpu_compiled.graph}")

                if stage == FxStage.pregrad:
                    return xpu_compiled

                if self._config.vendor_compiler_config:

                    from .backends import vendor_compiler

                    return vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        self._config.target,
                        self._config.vendor_compiler_config,
                    )

            return xpu_compiled

        def _staged_compiler(stage: FxStage):
            def wrapped(gm, sample_inputs):
                with local_logger(stage.name):
                    xpu_compiled = _compiler(gm, sample_inputs, stage)
                return xpu_compiled

            return wrapped

        if self._config.is_training:
            # Since: 1. dynamo has eliminated control-flow for input GraphModule
            #    and 2. aot_autograd traces grad again
            # It's okay use optimized infer-graph for training as well
            pregrad_gm = _staged_compiler(FxStage.pregrad)(dynamo_gm, example_inputs)

            xpu_gm = aot_autograd(
                fw_compiler=_staged_compiler(FxStage.forward),
                bw_compiler=_staged_compiler(FxStage.backward),
            )(pregrad_gm, example_inputs)
        else:
            logger.info("aot_export_module start...")
            logger.debug(f"before aot_export_module, graph like:\n {dynamo_gm.graph}")

            exported_gm, gs = aot_export_module(
                dynamo_gm, example_inputs, trace_joint=False
            )
            logger.info("aot_export_module complete")
            logger.debug(f"after aot_export_module, graph like:\n {exported_gm.graph}")
            logger.debug(f"graph signature: {gs}")

            logger.info("unlift graph start...")
            logger.debug(f"before unlift, graph like:\n {exported_gm.graph}")
            unlifted_gm = unlift_exported_gm(
                dynamo_gm, exported_gm, gs, freeze=self._config.freeze
            )
            logger.info("unlift graph complete")
            logger.debug(f"after unlift, graph like:\n {unlifted_gm.graph}")

            xpu_gm = _staged_compiler(FxStage.inference)(unlifted_gm, example_inputs)

        return xpu_gm

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()

    def _set_context(self):
        self._orig_ctx = {}
        self._orig_ctx["torch._inductor.config.freezing"] = (
            torch._inductor.config.freezing
        )
        if self._config.freeze and self._config.is_training == False:
            # The configuration in this inductor affects the return value of is_parameter_freezing(),
            # thereby influencing the process of generating the fx_graph in dynamo. The current code
            # in the community is not very clean, and it would be more reasonable to place this
            # configuration under dynamo. You can refer to this link for more information.
            # https://github.com/pytorch/pytorch/blob/release/2.5/torch/_dynamo/utils.py#L3061
            torch._inductor.config.freezing = True
        else:
            torch._inductor.config.freezing = False

        if self._cache is not None:
            self._orig_ctx["self._cache.orig_ctx"] = self._cache._set_cache_ctx()

    def _restore_context(self):
        torch._inductor.config.freezing = self._orig_ctx[
            "torch._inductor.config.freezing"
        ]
        if self._cache is not None:
            self._cache._restore_cache_ctx(self._orig_ctx["self._cache.orig_ctx"])
