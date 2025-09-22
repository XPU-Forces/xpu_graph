import os
from itertools import chain

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import FakeTensorMode

from .cache import SerializeWrapper, XpuGraphCache, default_cache
from .config import OptLevel, Target, XpuGraphConfig
from .fx_utils import FxStage, dispatch_graph
from .passes.pass_manager import PassManager
from .passes.patterns.plugin_pattern import __PLUGIN_PATTERN_GROUP__
from .utils import GitLikeDiffer, NodesStatistics, local_logger, logger, setup_logger

__all__ = [
    "optimize_graph",
    "XpuGraph",
]


class XpuGraph:
    def __init__(
        self,
        config: XpuGraphConfig,
        cache: XpuGraphCache = None,
    ):
        config._reset_config_with_env()
        self._config = config
        # Setup logging based on config
        setup_logger(self._config.debug)

        logger.info(f"{config}")

        if self._config.target == Target.npu and self._config.vendor_compiler_config.get("compiler", None) == "ge":
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
                original_gm_graph = str(gm.graph)
                with local_logger("before"):
                    logger.debug(f"before xpu_graph, graph like:\n {original_gm_graph}")
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

                if stage != FxStage.pregrad and self._config.vendor_compiler_config is not None:
                    from .backends import vendor_compiler

                    xpu_compiled = vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        target=self._config.target,
                        is_inference=stage == FxStage.inference,
                        is_backward=stage == FxStage.backward,
                        **self._config.vendor_compiler_config,
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
            dispatched_gm, fake_inputs, fw_metadata = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.pregrad)
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
            dispatched_gm, fake_inputs, fw_metadata = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.inference)
            logger.info("decompose graph complete")
            logger.debug(f"after decompose, graph like:\n {dispatched_gm.graph}")

            xpu_gm = _staged_compiler(FxStage.inference)(dispatched_gm, fake_inputs)

        if self._config.enable_interceptor is not None:
            from xpu_graph.interceptor import intercept

            logger.info("Wrapping compiled funciton with interceptor")
            xpu_gm = intercept(
                xpu_gm,
                golden_fn=dynamo_gm,
                fw_metadata=fw_metadata,
                is_training=self._config.is_training,
                mark=os.environ.get("RANK", "0"),
                config_str=self._config.enable_interceptor,
            )
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

    def aot_export(
        self,
        mod: torch.nn.Module,
        example_args,
        example_kwargs=None,
        package_path=None,
        *args,
        **kwargs,
    ):
        from packaging import version

        torch_version = version.parse(torch.__version__[:5])
        if torch_version < version.parse("2.6.0"):
            logger.error("AOT export functionality is only available on torch 2.6 for now")
            raise NotImplemented
        if self._config.is_training:
            logger.error("AOT export functionality is only available for inference")
            raise NotImplemented

        example_kwargs = example_kwargs or {}

        logger.info("export module start...")
        exported_prog = torch.export.export(mod, example_args, example_kwargs, strict=False)
        logger.info("export module complete")

        flat_inputs = exported_prog._graph_module_flat_inputs(*exported_prog.example_inputs)
        compiled_func = self(exported_prog._graph_module, flat_inputs)
        optimized_mod = OptimizedModule(exported_prog, compiled_func)
        return optimized_mod


class OptimizedModule(torch.nn.Module):
    def __init__(self, exported_program, optimized_gm):
        super().__init__()
        self.exported_program = exported_program
        self.optimized_gm = optimized_gm

    def forward(self, *args, **kwargs):
        flat_inputs = self.exported_program._graph_module_flat_inputs(args, kwargs)
        flat_outputs = self.optimized_gm(*flat_inputs)
        return self.exported_program._postprocess_graph_module_outputs(flat_outputs, args, kwargs)
