import copy
import hashlib
import importlib
import os
import pickle
from os import PathLike
from typing import Union

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


def serialize_artifact(artifact):
    if isinstance(artifact, GraphModule):
        return GmSerializeHelper.serialize_fn(artifact)
    elif (serialize_fn := getattr(artifact, "_xpugraph_serialize_fn", None)) is not None:
        # Note: this is inspired by
        return serialize_fn(artifact)
    else:
        raise NotImplementedError(f"Unsupported serialization for {type(artifact)}")


def deserialize_artifact(serialized):
    deserialize_fn, deserialize_args = serialized
    return deserialize_fn(deserialize_args)


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
        try:
            serialized = serialize_artifact(value)
        except NotImplementedError:
            logger.warning(f"Cannot serialize type {type(value)}, skip: {value}")
            return value
        artifact_path = self._artifact_path(key)
        logger.info(f"Save cache in location: {artifact_path}")
        with compile_lock, _disable_current_modes(), TracingContext.patch(fake_mode=None):
            with open(artifact_path, "wb+") as f:
                pickle.dump(serialized, f)
            with open(artifact_path, "rb") as f:
                deserialize_fn, deserialize_args = pickle.load(f)
                cached_graph = deserialize_fn(deserialize_args)
        return cached_graph

    def load_artifact(self, key):
        artifact_path = self._artifact_path(key)
        if os.path.isfile(artifact_path):
            with compile_lock, _disable_current_modes(), TracingContext.patch(fake_mode=None):
                logger.info(f"Use cache in location: {artifact_path}")
                with open(artifact_path, "rb") as f:
                    deserialize_fn, deserialize_args = pickle.load(f)
            return deserialize_fn(deserialize_args)
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
