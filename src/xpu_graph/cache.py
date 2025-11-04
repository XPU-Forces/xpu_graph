import hashlib
import importlib
import os
import pickle
from abc import ABC, abstractclassmethod, abstractmethod
from functools import wraps
from os import PathLike
from typing import Union
from contextlib import contextmanager, nullcontext

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._dynamo.device_interface import get_interface_for_device
from torch._guards import TracingContext
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_aggregate
from torch.utils._python_dispatch import _disable_current_modes

from .config import Target, XpuGraphConfig, get_cache_dir
from .fx_utils import FxStage
from .utils import PolyBackendDispatcher, logger


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
        if isinstance(artifact, SerializableArtifact):
            return
        super().__init__()
        assert callable(artifact), f"artifact must be callable, but got {type(artifact)}"
        self._artifact = artifact
        if getattr(artifact, "_boxed_call", False):
            self._boxed_call = True

    def __call__(self, *args, **kwargs):
        return self._artifact(*args, **kwargs)

    # NOTE(liuyuan): allow implicit no-conversion between subclasses of Serializable.
    def __new__(cls, artifact):
        if isinstance(artifact, SerializableArtifact):
            return artifact
        else:
            return super().__new__(cls)

    @property
    def artifact(self):
        return self._artifact

    def __reduce__(self):
        return self.rebuild_from_bytes, (self.convert_to_bytes(),)

    @abstractmethod
    def convert_to_bytes(self) -> bytes:
        # TODO(liuyuan): For performance, try to make it as a zero copy byte strings.
        """
        Convert artifact to bytes. The return value should be artifact_bytes,
        which can rebuild the artifact via rebuild_from_bytes.
        """
        ...

    @staticmethod
    @abstractmethod
    def rebuild_from_bytes(byte_s: bytes):
        """
        Rebuild artifact from bytes. The input is the artifact_bytes from convert_to_bytes.
        """
        ...


@contextmanager
def temp_disable_tracing_envs():
    with compile_lock, _disable_current_modes():
        tracing_context = TracingContext.try_get()
        with tracing_context.patch(fake_mode=None) if tracing_context else nullcontext():
            yield

class SerializableGraphModule(SerializableArtifact):
    def __init__(self, artifact):
        assert isinstance(artifact, GraphModule)
        super().__init__(artifact)

    def convert_to_bytes(self) -> bytes:
        with temp_disable_tracing_envs():
            gm_dict, graph_meta, nodes_meta = GmSerializeHelper.serialize_fn(self._artifact)
            return pickle.dumps((gm_dict, graph_meta, nodes_meta))

    @staticmethod
    def rebuild_from_bytes(byte_s: bytes):
        with temp_disable_tracing_envs():
            gm_dict, graph_meta, nodes_meta = pickle.loads(byte_s)
            gm = GmSerializeHelper.deserialize_fn(gm_dict, graph_meta, nodes_meta)
            return __class__(gm)


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

        return gm_dict, graph_meta, nodes_meta

    @staticmethod
    def deserialize_fn(gm_dict, graph_meta, nodes_meta):
        for k, v in gm_dict["_modules"].items():
            if isinstance(v, tuple):
                gm_dict["_modules"][k] = GmSerializeHelper.deserialize_fn(*v)
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


class SerializableCompiledFxGraph(SerializableArtifact):
    def __init__(self, artifact):
        from torch._inductor.compile_fx import CompiledFxGraph

        assert isinstance(artifact, CompiledFxGraph)
        super().__init__(artifact)

    def convert_to_bytes(self) -> bytes:
        with temp_disable_tracing_envs():
            current_callable = self._artifact.current_callable
            self._artifact.current_callable = None
            serialized = pickle.dumps(self._artifact)
            self._artifact.current_callable = current_callable
            return serialized

    @staticmethod
    def rebuild_from_bytes(byte_s):
        from torch._inductor.codecache import FxGraphCache, PyCodeCache, get_path
        from torch._inductor.compile_fx import CompiledFxGraph
        from torch._inductor.utils import BoxedBool

        logger.info(f"Deserializing a {CompiledFxGraph.__qualname__}")
        with temp_disable_tracing_envs():
            # Disable tracing envs is necessary because we need to load REAL tensors as parameters and buffers
            compiled_fn = pickle.loads(byte_s)
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
        torch_version = torch.__version__[:3]
        if torch_version.startswith("2.5"):
            tracing_context = torch._guards.TracingContext.try_get()
            if tracing_context is not None and tracing_context.output_strides:
                tracing_context.output_strides.clear()
            FxGraphCache.post_compile(compiled_fn, example_inputs=[], cudagraphs=BoxedBool(cudagraphs))
        return __class__(compiled_fn)


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
        with open(artifact_path, "wb+") as f:
            pickle.dump(value, f)
        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    def load_artifact(self, key):
        artifact_path = self._artifact_path(key)
        if os.path.isfile(artifact_path):
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
