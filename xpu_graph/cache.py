import copy
import hashlib
import os
import pickle
import sys
from os import PathLike
from typing import Optional, Union

import torch
from torch._dynamo.convert_frame import compile_lock
from torch._dynamo.device_interface import get_interface_for_device
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

from collections.abc import Callable

from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_aggregate
from torch.utils._python_dispatch import _disable_current_modes

from .config import XpuGraphConfig
from .fx_utils import FxStage
from .utils import logger


class _ArgWrapper:
    """Helper function for storing fx.Node arg that are nodes"""

    def __init__(self, n: Node):
        self.name = n.name


def _get_target_function(fn_name: str):
    fqn_list = fn_name.split(".")
    try:
        target = sys.modules[fqn_list[0]]
        for attr in fqn_list[1:]:
            target = getattr(target, attr)
        assert callable(target)
    except:
        raise NotImplementedError(f"Unsupported call_function: {fn_name}")
    return target


class SerializeWrapper(torch.nn.Module):
    def __init__(self, compiled_fn: Union[CompiledFxGraph, GraphModule, Callable]):
        super().__init__()
        self.wrapped_fn = compiled_fn

    def __reduce__(self):
        return (
            SerializeWrapper.deserialize_helper,
            SerializeWrapper.serialize_helper(self.wrapped_fn),
        )

    def forward(self, *runtime_args):
        if hasattr(self.wrapped_fn, "_boxed_call") and self.wrapped_fn._boxed_call:
            # Note: if the wrapped_fn is a boxed function, unbox it now
            args = []
            args.extend(runtime_args)
            return self.wrapped_fn(args)
        return self.wrapped_fn(*runtime_args)

    @staticmethod
    def serialize_helper(object):
        if isinstance(object, CompiledFxGraph):
            mod = copy.copy(object)
            mod.current_callable = None
            return (CompiledFxGraph, (mod,))
        elif isinstance(object, GraphModule):
            gm_dict = object.__dict__.copy()
            del gm_dict["_graph"]
            for k, v in gm_dict["_modules"].items():
                if isinstance(v, GraphModule):
                    gm_dict["_modules"][k] = SerializeWrapper(v)
            graph = object.graph
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

            return (GraphModule, (gm_dict, graph_meta, nodes_meta))
        else:
            raise NotImplemented(f"Unsupported type: {type(object)} for {object}")

    def deserialize_helper(cls, arg_tuple):
        logger.info(f"Deserializing a {cls.__qualname__}")
        if cls == CompiledFxGraph:
            (compiled_fn,) = arg_tuple
            # Torch Inductor config is lazy initialized. invoke it manually
            for device in compiled_fn.device_types:
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
                FxGraphCache.post_compile(compiled_fn, example_inputs=[], cudagraphs=BoxedBool(cudagraphs))
            return SerializeWrapper(compiled_fn)
        elif cls == GraphModule:
            gm_dict, graph_meta, nodes_meta = arg_tuple
            for k, v in gm_dict["_modules"].items():
                if isinstance(v, SerializeWrapper):
                    gm_dict["_modules"][k] = v.wrapped_fn
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
            return SerializeWrapper(gm)
        else:
            raise NotImplementedError(f"Unsupported deserialize: {cls}, {arg_tuple}")


class XpuGraphCache:
    """A base cache class does not store any thing"""

    def cache_key(
        self,
        gm: torch.fx.GraphModule,
        fake_inputs,
        config: XpuGraphConfig,
        stage: FxStage,
    ):
        key = f"{gm}-{fake_inputs}-{config}-{stage}"
        logger.debug(f"Cache Key readable: \n{key}")
        hashkey = hashlib.md5(key.encode()).hexdigest()
        logger.info(f"Cache Key: {hashkey}")
        return hashkey

    def save_gm(self, key, value: SerializeWrapper, expire=None) -> SerializeWrapper:
        # Note: since GraphModules ser/des may do canonicalization, so the cached version should be returned
        return value

    def load_gm(self, key) -> Optional[SerializeWrapper]:
        return None

    def delete_gm(self, key):
        return None

    def _set_cache_ctx(self):
        return None

    def _restore_cache_ctx(self, orig_ctx):
        pass


class XpuGraphLocalCache(XpuGraphCache):
    def __init__(self, cache_path: PathLike):
        super().__init__()
        cache_path = os.path.abspath(cache_path)
        os.makedirs(cache_path, exist_ok=True)
        self._path = cache_path

    def save_gm(self, key, value: SerializeWrapper, expire=None) -> SerializeWrapper:
        artifact_path = self._graph_path(key)
        logger.info(f"Save cache in location: {artifact_path}")
        with compile_lock, _disable_current_modes():
            with open(artifact_path, "wb+") as f:
                pickle.dump(value, f)
            with open(artifact_path, "rb") as f:
                cached_graph = pickle.load(f)
        return cached_graph

    def load_gm(self, key) -> Optional[SerializeWrapper]:
        artifact_path = self._graph_path(key)
        if os.path.isfile(artifact_path):
            with compile_lock, _disable_current_modes():
                logger.info(f"Use cache in location: {artifact_path}")
                with open(artifact_path, "rb") as f:
                    cached_graph = pickle.load(f)
            return cached_graph
        else:
            return None

    def delete_gm(self, key):
        if key in self.cache:
            del self.cache[key]

    def _graph_path(self, key):
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
    cache_path = os.getenv("XPUGRAPH_CACHE_DIR")
    if cache_path is None:
        import tempfile

        cache_path = tempfile.mkdtemp(prefix="xpugraph_")
        os.environ["XPUGRAPH_CACHE_DIR"] = cache_path
        logger.debug(f"Use {cache_path} as default local cache")
    return XpuGraphLocalCache(cache_path)
