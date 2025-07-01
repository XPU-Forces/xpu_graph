import logging
from itertools import chain

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import FakeTensorMode

from .cache import SerializeWrapper, XpuGraphCache, default_cache
from .config import OptLevel, Target, XpuGraphConfig
from .fx_utils import FxStage, decompose_for_inductor, dispatch_graph
from .passes.pass_manager import PassManager
from .passes.patterns.plugin_pattern import __PLUGIN_PATTERN_GROUP__
from .utils import GitLikeDiffer, NodesStatistics, local_logger, logger, setup_logger

__all__ = [
    "optimize_graph",
    "XpuGraph",
]


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
    fake_inputs = [fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in sample_inputs]

    with fake_mode:
        logger.debug(f"before xpu_optimize_graph, graph like:\n {gm.graph}")
        logger.info(f"before xpu_optimize_graph, nodes num: {len(gm.graph.nodes)}")

        pass_manager = PassManager(config)
        xpu_optimized = pass_manager(gm, fake_inputs, stage=FxStage.inference)

        logger.debug(f"after xpu_optimize_graph, graph like:\n {xpu_optimized.graph}")
        logger.info(f"after xpu_optimize_graph, nodes num: {len(xpu_optimized.graph.nodes)}")

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

        if self._config.target == Target.npu and self._config.vendor_compiler_config["compiler"] == "ge":
            self._config.enable_cache = False
            logger.warning("Target NPU ge-compiler does not support cache.")

        self._cache = cache if cache and config.enable_cache else default_cache() if config.enable_cache else None
        self._set_context()
        # WARNING(liuyuan): _pass_manager MUST be initilized after _set_context because triton kernel depends on environment varaibels that fetched in _set_context.
        self._pass_manager = PassManager(self._config)
        # NOTE(liuyuan): The plugin patterns should be placed before those built-in.
        self._pass_manager.get_pattern_manager().insert_patterns(
            chain.from_iterable(__PLUGIN_PATTERN_GROUP__.get(self._config.target, {}).values())
        )

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        def _compiler(gm, fake_inputs, stage: FxStage):
            nodes_statistics = NodesStatistics()

            # Create fake inputs for optimization
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(fake_inputs)
            fake_mode.allow_non_fake_inputs = True

            with fake_mode:
                if self._config.enable_cache:
                    hashkey = self._cache.cache_key(gm, fake_inputs, self._config, stage)
                    cached_compiled = self._cache.load_gm(hashkey)
                    if cached_compiled is not None:
                        return cached_compiled

                # NOTE(liuyuan): gm could be changed in the compiler, and we should keep the original graph for logging difference.
                original_gm_graph = gm.graph
                with local_logger("before"):
                    logger.debug(f"before xpu_graph, graph like:\n {gm.graph}")
                    logger.info(f"xpu_graph passes start {stage}...")

                nodes_statistics.insert_statistics("before xpu_graph", gm)
                xpu_compiled = self._pass_manager(gm, fake_inputs, stage)
                nodes_statistics.insert_statistics("after xpu_graph", xpu_compiled)

                with local_logger("after"):
                    logger.info("xpu_graph passes complete")
                    logger.debug(f"after xpu_graph, graph like:\n {xpu_compiled.graph}")
                    logger.debug(
                        "Final difference after optimizations by xpu_graph:%s\n",
                        GitLikeDiffer(original_gm_graph, xpu_compiled.graph),
                    )

                logger.info(f"node statistic: {str(nodes_statistics)}")

                if stage != FxStage.pregrad and self._config.vendor_compiler_config:
                    xpu_compiled = decompose_for_inductor(xpu_compiled, fake_inputs)
                    extra_kwargs = {}
                    if stage == FxStage.inference:
                        extra_kwargs["is_inference"] = True
                    elif stage == FxStage.backward:
                        extra_kwargs["is_backward"] = True
                    from .backends import vendor_compiler

                    xpu_compiled = vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        self._config.target,
                        self._config.vendor_compiler_config,
                        **extra_kwargs,
                    )

                xpu_compiled = SerializeWrapper(xpu_compiled)

                if self._config.enable_cache:
                    xpu_compiled = self._cache.save_gm(hashkey, xpu_compiled)

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
            logger.debug(f"before decompose: graph like:\n {dynamo_gm.graph}")
            logger.info("decompose graph start...")
            dispatched_gm, fake_inputs = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.pregrad)
            logger.info("decompose graph complete")
            logger.debug(f"after decompose, graph like:\n {dispatched_gm.graph}")

            pregrad_gm = _staged_compiler(FxStage.pregrad)(dispatched_gm, fake_inputs)

            xpu_gm = aot_autograd(
                fw_compiler=_staged_compiler(FxStage.forward),
                bw_compiler=_staged_compiler(FxStage.backward),
            )(pregrad_gm, fake_inputs)
        else:
            logger.debug(f"before decompose: graph like:\n {dynamo_gm.graph}")
            logger.info("decompose graph start...")
            dispatched_gm, fake_inputs = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.inference)
            logger.info("decompose graph complete")
            logger.debug(f"after decompose, graph like:\n {dispatched_gm.graph}")

            xpu_gm = _staged_compiler(FxStage.inference)(dispatched_gm, fake_inputs)

        return xpu_gm

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()

    def _set_context(self):
        self._orig_ctx = {}
        self._orig_ctx["torch._inductor.config.freezing"] = torch._inductor.config.freezing
        if self._config.freeze and self._config.is_training == False:
            # The configuration in this inductor affects the return value of is_parameter_freezing(),
            # thereby influencing the process of generating the fx_graph in dynamo. The current code
            # in the community is not very clean, and it would be more reasonable to place this
            # configuration under dynamo. You can refer to this link for more information.
            # https://github.com/pytorch/pytorch/blob/release/2.5/torch/_dynamo/utils.py#L3061
            torch._inductor.config.freezing = True
        else:
            torch._inductor.config.freezing = False

        if self._config.target != Target.none:
            if torch._dynamo.config.trace_numpy:
                self._orig_ctx["torch._dynamo.config.numpy_default_float"] = torch._dynamo.config.numpy_default_float
                logger.info("xpu_graph set the default traced numpy float dtype to float32")
                torch._dynamo.config.numpy_default_float = "float32"

        if self._cache is not None:
            self._orig_ctx["self._cache.orig_ctx"] = self._cache._set_cache_ctx()

    def _restore_context(self):
        torch._inductor.config.freezing = self._orig_ctx["torch._inductor.config.freezing"]
        if "torch._dynamo.config.numpy_default_float" in self._orig_ctx:
            torch._dynamo.config.numpy_default_float = self._orig_ctx["torch._dynamo.config.numpy_default_float"]
        if self._cache is not None:
            self._cache._restore_cache_ctx(self._orig_ctx["self._cache.orig_ctx"])
