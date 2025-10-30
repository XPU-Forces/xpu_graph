import functools
import os
from itertools import chain

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import create_aot_dispatcher_function
from torch._subclasses.fake_tensor import FakeTensorMode

from .cache import SerializeWrapper, XpuGraphCache, default_cache
from .config import OptLevel, Target, XpuGraphConfig, get_partition_fn
from .fx_utils import (
    FxStage,
    dispatch_graph,
    fakify_tensors,
    find_hop_nodes,
    find_subclass_inputs,
    freeze_compile,
)
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
    setup_logger(config.debug)

    logger.info(f"{config}")

    # Create fake inputs for optimization
    fake_mode = FakeTensorMode()
    fake_mode.allow_non_fake_inputs = True
    fake_inputs = [fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in sample_inputs]

    with fake_mode:
        logger.debug(
            "before xpu_optimize_graph, graph like:\n %s",
            gm.print_readable(print_output=False, include_stride=True, include_device=True),
        )
        logger.info(f"before xpu_optimize_graph, nodes num: {len(gm.graph.nodes)}")

        pass_manager = PassManager(config)
        xpu_optimized = pass_manager(gm, fake_inputs, stage=FxStage.inference)

        logger.debug(
            "after xpu_optimize_graph, graph like:\n %s",
            xpu_optimized.print_readable(print_output=False, include_stride=True, include_device=True),
        )
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
        setup_logger(self._config.debug)

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

    def _get_compiler(self, stage: FxStage):
        def _compiler(gm, fake_inputs, stage: FxStage):
            nodes_statistics = NodesStatistics()

            # Create fake inputs for optimization
            fake_mode, fake_inputs = fakify_tensors(fake_inputs)

            with fake_mode:
                if self._config.enable_cache:
                    hashkey = self._cache.cache_key(gm, fake_inputs, self._config, stage)
                    cached_compiled = self._cache.load_gm(hashkey)
                    if cached_compiled is not None:
                        return cached_compiled

                # NOTE(liuyuan): gm could be changed in the compiler, and we should keep the original graph for logging difference.
                original_gm_graph = gm.print_readable(print_output=False, include_stride=True, include_device=True)
                with local_logger("before"):
                    logger.debug(f"before xpu_graph, graph like:\n {original_gm_graph}")
                    logger.info(f"xpu_graph passes start {stage}...")

                nodes_statistics.insert_statistics("before xpu_graph", gm)
                xpu_compiled = self._pass_manager(gm, fake_inputs, stage)
                nodes_statistics.insert_statistics("after xpu_graph", xpu_compiled)

                compiled_gm_graph = xpu_compiled.print_readable(
                    print_output=False, include_stride=True, include_device=True
                )
                with local_logger("after"):
                    logger.info("xpu_graph passes complete")
                    logger.debug(f"after xpu_graph, graph like:\n {compiled_gm_graph}")
                    logger.debug(
                        "Final difference after optimizations by xpu_graph:%s\n",
                        GitLikeDiffer(original_gm_graph, compiled_gm_graph),
                    )

                logger.info(f"node statistic: {str(nodes_statistics)}")

                if (
                    stage in [FxStage.inference, FxStage.forward, FxStage.backward]
                    and self._config.vendor_compiler_config is not None
                ):
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

                    # if self._config.enable_cache:
                    #     xpu_compiled = self._cache.save_gm(hashkey, xpu_compiled)

            return xpu_compiled

        def wrapped(gm, sample_inputs):
            with local_logger(stage.name):
                xpu_compiled = _compiler(gm, sample_inputs, stage)
            return xpu_compiled

        return wrapped

    def _legacy_dispatch_and_compile(self, dynamo_gm, example_inputs):
        fallback_dispatch = False
        if self._config.fallback_legacy_dispatch:
            if self._config.is_training:
                hop_nodes = find_hop_nodes(dynamo_gm)
                if len(hop_nodes) > 0:
                    logger.warning(f"Higher order operators detected: {', '.join(str(op) for op in hop_nodes)}.")
                    fallback_dispatch = True

        subclass_tensors = find_subclass_inputs(example_inputs)
        if len(subclass_tensors) > 0:
            logger.warning(f"Subclass inputs detected: {', '.join(str(cls) for cls in subclass_tensors)}.")
            fallback_dispatch = True

        if fallback_dispatch:
            logger.debug(
                "before compile: graph like:\n %s",
                dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )
            kwargs = {}
            if self._config.is_training:
                kwargs["fw_compiler"] = self._get_compiler(FxStage.forward)
                kwargs["bw_compiler"] = self._get_compiler(FxStage.backward)
                kwargs["keep_inference_input_mutations"] = True
                if partition_fn := get_partition_fn(self._config.partition_fn):
                    kwargs["partition_fn"] = partition_fn
            else:
                kwargs["fw_compiler"] = self._get_compiler(FxStage.inference)
                kwargs["keep_inference_input_mutations"] = True
            xpu_gm = aot_autograd(**kwargs)(dynamo_gm, example_inputs)
            fw_metadata = None
        elif self._config.is_training:
            # Since: 1. dynamo has eliminated control-flow for input GraphModule
            #    and 2. aot_autograd traces grad again
            # It's okay use optimized infer-graph for training as well
            logger.debug(
                "before decompose: graph like:\n %s",
                dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )
            logger.info("decompose graph start...")
            dispatched_gm, fake_inputs, fw_metadata = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.pregrad)
            logger.info("decompose graph complete")
            logger.debug(
                "after decompose, graph like:\n %s",
                dispatched_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )

            pregrad_gm = self._get_compiler(FxStage.pregrad)(dispatched_gm, fake_inputs)

            kwargs = {}
            kwargs["fw_compiler"] = self._get_compiler(FxStage.forward)
            kwargs["bw_compiler"] = self._get_compiler(FxStage.backward)
            if partition_fn := get_partition_fn(self._config.partition_fn):
                kwargs["partition_fn"] = partition_fn
            xpu_gm = aot_autograd(**kwargs)(pregrad_gm, fake_inputs)

        else:
            logger.debug(
                "before decompose: graph like:\n %s",
                dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )
            logger.info("decompose graph start...")
            dispatched_gm, fake_inputs, fw_metadata = dispatch_graph(dynamo_gm, example_inputs, stage=FxStage.inference)
            logger.info("decompose graph complete")
            logger.debug(
                "after decompose, graph like:\n %s",
                dispatched_gm.print_readable(print_output=False, include_stride=True, include_device=True),
            )

            xpu_gm = self._get_compiler(FxStage.inference)(dispatched_gm, fake_inputs)

        return xpu_gm, fw_metadata

    def _dispatch_and_compile(self, dynamo_gm, example_inputs, *args, **kwargs):
        logger.debug(
            "before compile: graph like:\n %s",
            dynamo_gm.print_readable(print_output=False, include_stride=True, include_device=True),
        )
        kwargs = {}
        kwargs["keep_inference_input_mutations"] = True
        kwargs["fw_compiler"] = self._get_compiler(FxStage.forward)
        kwargs["bw_compiler"] = self._get_compiler(FxStage.backward)

        def partition_fn(joint_gm, joint_args, *, num_fwd_outputs):
            new_joint_gm = self._get_compiler(FxStage.joint)(joint_gm, joint_args)

            from torch._functorch.partitioners import default_partition

            partition_fn = get_partition_fn(self._config.partition_fn) or default_partition

            return partition_fn(new_joint_gm, joint_args, num_fwd_outputs=num_fwd_outputs)

        kwargs["partition_fn"] = partition_fn
        if self._config.freeze:
            kwargs["inference_compiler"] = functools.partial(
                freeze_compile, dynamo_gm, inner_compiler=self._get_compiler(FxStage.inference)
            )
        else:
            kwargs["inference_compiler"] = self._get_compiler(FxStage.inference)

        compiled = aot_autograd(**kwargs)(dynamo_gm, example_inputs)

        if tracing_context := torch._guards.TracingContext.try_get():
            fw_metadata = tracing_context.fw_metadata
        else:
            fw_metadata = None

        return compiled, fw_metadata

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        logger.info(f"{self._config}")

        if self._config.fallback_legacy_dispatch:
            xpu_gm, fw_metadata = self._legacy_dispatch_and_compile(dynamo_gm, example_inputs)
        else:
            xpu_gm, fw_metadata = self._dispatch_and_compile(dynamo_gm, example_inputs)

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
