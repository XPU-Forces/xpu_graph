import copy
import hashlib
import importlib
import os
import pickle
from abc import ABC, abstractmethod, abstractstaticmethod
from os import PathLike
from typing import Callable, Dict, Tuple, Union

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._dynamo.device_interface import get_interface_for_device
from torch._guards import TracingContext
from torch._inductor.utils import BoxedBool
from torch.utils._python_dispatch import _disable_current_modes

torch_version = torch.__version__[:3]
if torch_version >= "2.6":
    from torch._inductor.codecache import FxGraphCache, PyCodeCache, get_path
    from torch._inductor.output_code import CompiledFxGraph
else:
    from torch._inductor.codecache import (
        CompiledFxGraph,
        PyCodeCache,
        get_path,
        FxGraphCache,
    )

from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_aggregate

from .config import XpuGraphConfig, get_cache_dir
from .fx_utils import FxStage
from .utils import logger


class _ArgWrapper:
    """Helper function for storing fx.Node arg that are nodes"""

    def __init__(self, n: Node):
        self.name = n.name


def _get_target_function(fn_name: str):
    fqn_list = fn_name.split(".")
    try:
        target = importlib.import_module(fqn_list[0])
        for attr in fqn_list[1:]:
            target = getattr(target, attr)
        assert callable(target)
    except:
        raise NotImplementedError(f"Unsupported call_function: {fn_name}")
    return target


class SerializableArtifact(ABC):
    def __init__(self, artifact):
        super().__init__()
        self._artifact = artifact

    @property
    def artifact(self):
        return self._artifact

    def __reduce__(self):
        return (self._deserialize, self._serialize())

    @abstractmethod
    def _serialize(self) -> Tuple:
        """
        The return tuple will be passed to __deserializa function.
        """
        ...

    @abstractstaticmethod
    def _deserialize(*args):
        ...


class SerializableGraphModule(SerializableArtifact):
    def __init__(self, artifact):
        assert isinstance(artifact, GraphModule)
        super().__init__(artifact)

    def _serialize(self) -> Tuple:
        gm_dict = self._artifact.__dict__.copy()
        del gm_dict["_graph"]
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, GraphModule):
                gm_dict["_modules"][k] = GmSerializeHelper.serialize_fn(v)
        graph = self._artifact.graph
        graph_meta = (graph._tracer_cls, graph._tracer_extras)
        nodes = list(graph.nodes)
        nodes_meta = []

        def _wrap_arg(arg):
            if isinstance(arg, Node):
                return _ArgWrapper(arg)
            else:
                return arg

        for node in nodes:
            node_meta = (
                node.name,
                node.type,
                node.op,
                node._pretty_print_target(node.target),
                tuple(map_aggregate(node.args, _wrap_arg)),
                dict(map_aggregate(node.kwargs, _wrap_arg)),
            )
            nodes_meta.append(node_meta)
        return (gm_dict, graph_meta, nodes_meta)

    @staticmethod
    def _deserialize(gm_dict, graph_meta, nodes_meta):
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, tuple) and v[0] == GmSerializeHelper.deserialize_fn:
                des_fn, deserialize_args = v
                gm_dict["_modules"][k] = des_fn(deserialize_args)
        gm = GraphModule.__new__(GraphModule)
        gm.__dict__ = gm_dict

        tracer_cls, tracer_extras = graph_meta
        graph = Graph(gm, tracer_cls, tracer_extras)

        _node_name_to_node = {}

        def _unwrap_arg(arg):
            if isinstance(arg, _ArgWrapper):
                return _node_name_to_node[arg.name]
            else:
                return arg

        for node_meta in nodes_meta:
            node_name, node_type, node_op, node_target, node_args, node_kwargs = node_meta

            if node_op == "call_function":
                node_target = _get_target_function(node_target)

            node_args = tuple(map_aggregate(node_args, _unwrap_arg))
            node_kwargs = dict(map_aggregate(node_kwargs, _unwrap_arg))
            _node_name_to_node[node_name] = graph.create_node(
                node_op, node_target, node_args, node_kwargs, node_name, node_type
            )
        gm.graph = graph
        gm.recompile()
        return gm


class SerializableFxGraph(SerializableArtifact):
    def __init__(self, artifact):
        assert isinstance(artifact, CompiledFxGraph)
        super().__init__(artifact)

    def _serialize(self):
        mod = copy.copy(self._artifact)
        mod.current_callable = None
        return (mod,)

    @staticmethod
    def _deserialize(mod):
        logger.info(f"Deserializing a {CompiledFxGraph.__qualname__}")
        compiled_fn = mod
        # Torch Inductor config is lazy initialized. invoke it manually
        for device in compiled_fn.device_types:
            if device == "cpu":
                continue
            logger.debug(f"Check interface for device: {device}")
            get_interface_for_device(device)
        path = get_path(compiled_fn.cache_key, "py")[2]
        compiled_fn.current_callable = PyCodeCache.load_by_key_path(
            compiled_fn.cache_key,
            path,
            compiled_fn.cache_linemap,
            compiled_fn.constants,
        ).call
        cudagraphs = compiled_fn.cudagraph_info is not None
        logger.debug(f"Cudagraphs enabled: {cudagraphs}")
        # Note:
        #   1. This post_compile function is only available on 2.5.x,
        #      it may be in different locations in other versions
        #   2. Example_inputs in post_compile actually leads to symint guards,
        #      but we choose to not produce extra guards
        if torch_version.startswith("2.5"):
            tracing_context = torch._guards.TracingContext.try_get()
            if tracing_context is not None and tracing_context.output_strides:
                tracing_context.output_strides.clear()
            FxGraphCache.post_compile(compiled_fn, example_inputs=[], cudagraphs=BoxedBool(cudagraphs))
        return compiled_fn


class GmSerializeHelper:
    """Note: this is a backported serializer / deserializer for class GraphModule"""

    @staticmethod
    def serialize_fn(gm: GraphModule):
        gm_dict = gm.__dict__.copy()
        del gm_dict["_graph"]
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, GraphModule):
                gm_dict["_modules"][k] = GmSerializeHelper.serialize_fn(v)
        graph = gm.graph
        graph_meta = (graph._tracer_cls, graph._tracer_extras)
        nodes = list(graph.nodes)
        nodes_meta = []

        def _wrap_arg(arg):
            if isinstance(arg, Node):
                return _ArgWrapper(arg)
            else:
                return arg

        for node in nodes:
            node_meta = (
                node.name,
                node.type,
                node.op,
                node._pretty_print_target(node.target),
                tuple(map_aggregate(node.args, _wrap_arg)),
                dict(map_aggregate(node.kwargs, _wrap_arg)),
            )
            nodes_meta.append(node_meta)

        return (__class__.deserialize_fn, (gm_dict, graph_meta, nodes_meta))

    @staticmethod
    def deserialize_fn(arg_tuple):
        gm_dict, graph_meta, nodes_meta = arg_tuple
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, tuple) and v[0] == GmSerializeHelper.deserialize_fn:
                des_fn, deserialize_args = v
                gm_dict["_modules"][k] = des_fn(deserialize_args)
        gm = GraphModule.__new__(GraphModule)
        gm.__dict__ = gm_dict

        tracer_cls, tracer_extras = graph_meta
        graph = Graph(gm, tracer_cls, tracer_extras)

        _node_name_to_node = {}

        def _unwrap_arg(arg):
            if isinstance(arg, _ArgWrapper):
                return _node_name_to_node[arg.name]
            else:
                return arg

        for node_meta in nodes_meta:
            node_name, node_type, node_op, node_target, node_args, node_kwargs = node_meta

            if node_op == "call_function":
                node_target = _get_target_function(node_target)

            node_args = tuple(map_aggregate(node_args, _unwrap_arg))
            node_kwargs = dict(map_aggregate(node_kwargs, _unwrap_arg))
            _node_name_to_node[node_name] = graph.create_node(
                node_op, node_target, node_args, node_kwargs, node_name, node_type
            )
        gm.graph = graph
        gm.recompile()
        return gm


class CompiledFxGraphSerializeHelper:
    @staticmethod
    def serialize_fn(compiled_fx: CompiledFxGraph):
        mod = copy.copy(compiled_fx)
        mod.current_callable = None
        return (__class__.deserialize_fn, (mod,))

    @staticmethod
    def deserialize_fn(arg_tuple):
        logger.info(f"Deserializing a {CompiledFxGraph.__qualname__}")
        (compiled_fn,) = arg_tuple
        # Torch Inductor config is lazy initialized. invoke it manually
        for device in compiled_fn.device_types:
            if device == "cpu":
                continue
            logger.debug(f"Check interface for device: {device}")
            get_interface_for_device(device)
        path = get_path(compiled_fn.cache_key, "py")[2]
        compiled_fn.current_callable = PyCodeCache.load_by_key_path(
            compiled_fn.cache_key,
            path,
            compiled_fn.cache_linemap,
            compiled_fn.constants,
        ).call
        cudagraphs = compiled_fn.cudagraph_info is not None
        logger.debug(f"Cudagraphs enabled: {cudagraphs}")
        # Note:
        #   1. This post_compile function is only available on 2.5.x,
        #      it may be in different locations in other versions
        #   2. Example_inputs in post_compile actually leads to symint guards,
        #      but we choose to not produce extra guards
        if torch_version.startswith("2.5"):
            tracing_context = torch._guards.TracingContext.try_get()
            if tracing_context is not None and tracing_context.output_strides:
                tracing_context.output_strides.clear()
            FxGraphCache.post_compile(compiled_fn, example_inputs=[], cudagraphs=BoxedBool(cudagraphs))
        return compiled_fn


class SerializableArtifact:
    """Base class for serializable artifact"""

    def __init__(self, artifact):
        self._artifact = artifact
        # The _boxed_call is necessary as it marks the calling convension of compiled artifact
        # and aot_dispatcher (outside xpu_graph inner_compile) need this flag to distinguish between boxed_call and unboxed call
        if getattr(self.artifact, "_boxed_call", False):
            self._boxed_call = True

    def __call__(self, *args, **kwargs):
        return self.artifact(*args, **kwargs)

    @property
    def artifact(self):
        return self._artifact

    @overload
    def serialize(self):
        raise NotImplementedError(f"Unsupported serialize artifact: {type(self.artifact)}")

    def __reduce__(self):
        #       A serializable artifact should implement a serialize fn
        #       1. The serialize_fn should return a tuple of (deserialize_fn, deserialize_args)
        #       2. The deserialize_fn should accept deserialize_args to reconstruct the artifact
        serialized = self.serialize()
        assert (
            isinstance(serialized, tuple) and len(serialized) == 2 and callable(serialized[0])
        ), f"serialize_fn  should return a tuple of (deserialize_fn, deserialize_args)"
        return (self.__class__.deserialize, (self.__class__, *serialized))

    @staticmethod
    def deserialize(cls, artifact_deserializer, artifact_deserialize_args):
        return cls(artifact_deserializer(artifact_deserialize_args))


class SerializableGM(SerializableArtifact):
    """Serializable artifact for GraphModule"""

    def serialize(self):
        return GmSerializeHelper.serialize_fn(self.artifact)


class SerializableCompiledFxGraph(SerializableArtifact):
    """Serializable artifact for CompiledFxGraph"""

    def serialize(self):
        return CompiledFxGraphSerializeHelper.serialize_fn(self.artifact)


class XpuGraphCache:
    """A base cache class does not store any thing"""

    def cache_key(
        self,
        gm: GraphModule,
        fake_inputs,
        config: XpuGraphConfig,
        stage: FxStage,
    ):
        gm_str = gm.print_readable(print_output=False, include_stride=True, include_device=True)
        key = f"{gm_str}-{fake_inputs}-{config}-{stage}"
        logger.debug(f"Cache Key readable: \n{key}")
        hashkey = hashlib.md5(key.encode()).hexdigest()
        logger.info(f"Cache Key: {hashkey}")
        return hashkey

    def save_artifact(self, key, value, expire=None):
        # Note: since GraphModules ser/des may do canonicalization, so the cached version should be returned
        return value

    def load_artifact(self, key):
        return None

    def delete_artifact(self, key):
        return None

    def _set_cache_ctx(self):
        return None

    def _restore_cache_ctx(self, orig_ctx):
        pass


class XpuGraphLocalCache(XpuGraphCache):
    def __init__(self, cache_path: Union[str, PathLike]):
        super().__init__()
        cache_path = os.path.abspath(cache_path)
        os.makedirs(cache_path, exist_ok=True)
        self._path = cache_path

    def save_artifact(self, key, value, expire=None):
        assert isinstance(value, SerializableArtifact)

        artifact_path = self._artifact_path(key)
        logger.info(f"Save cache in location: {artifact_path}")
        with compile_lock, _disable_current_modes(), TracingContext.patch(fake_mode=None):
            with open(artifact_path, "wb+") as f:
                pickle.dump(value, f)
            with open(artifact_path, "rb") as f:
                _ = pickle.load(f)

    def load_artifact(self, key):
        artifact_path = self._artifact_path(key)
        if os.path.isfile(artifact_path):
            with compile_lock, _disable_current_modes(), TracingContext.patch(fake_mode=None):
                logger.info(f"Use cache in location: {artifact_path}")
                with open(artifact_path, "rb") as f:
                    return pickle.load(f)
        else:
            return None

    def delete_artifact(self, key):
        if key in self.cache:
            del self.cache[key]

    def _artifact_path(self, key):
        fname = f"xpugraph_{key}.pt"
        artifact_cache = os.path.join(self._path, fname)
        return artifact_cache

    def _set_cache_ctx(self):
        orig_ctx = {}
        if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
            orig_ctx["TORCHINDUCTOR_CACHE_DIR"] = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        if "TRITON_CACHE_DIR" in os.environ:
            orig_ctx["TRITON_CACHE_DIR"] = os.environ["TRITON_CACHE_DIR"]

        # FIXME: Currently we manually set inductor cache dir for vendor compiler
        #        environs should not be tainted once AOT pipeline is ready
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(self._path, "inductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(self._path, "triton")
        return orig_ctx

    def _restore_cache_ctx(self, orig_ctx):
        if "TORCHINDUCTOR_CACHE_DIR" in orig_ctx:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = orig_ctx["TORCHINDUCTOR_CACHE_DIR"]
        else:
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
        if "TRITON_CACHE_DIR" in orig_ctx:
            os.environ["TRITON_CACHE_DIR"] = orig_ctx["TRITON_CACHE_DIR"]
        else:
            del os.environ["TRITON_CACHE_DIR"]


def no_cache():
    return XpuGraphCache()


def default_cache():
    cache_path = get_cache_dir()
    return XpuGraphLocalCache(cache_path)
