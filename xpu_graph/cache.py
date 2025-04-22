import copy
import hashlib
import os
import pickle
from os import PathLike

import torch
from torch._dynamo.convert_frame import compile_lock
from torch.utils._python_dispatch import _disable_current_modes

torch_version = torch.__version__
if torch_version.startswith("2.6"):
    from torch._inductor.compile_fx import CompiledFxGraph
    from torch._inductor.codecache import PyCodeCache, get_path
else:
    from torch._inductor.codecache import CompiledFxGraph, PyCodeCache, get_path
from torch.fx import GraphModule, Graph, Node
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
    import builtins
    import operator

    import torch

    supported_mods = {"torch": torch, "operator": operator, "builtins": builtins}
    try:
        target = supported_mods[fqn_list[0]]
        for attr in fqn_list[1:]:
            target = getattr(target, attr)
        assert callable(target)
    except:
        raise NotImplementedError(f"Unsupported call_function: {fn_name}")
    return target


class SerializeWrapper(torch.nn.Module):
    def __init__(self, compiled_fn):
        super().__init__()
        assert isinstance(compiled_fn, (CompiledFxGraph, GraphModule))
        self.wrapped_fn = compiled_fn

    def __reduce__(self):
        return (SerializeWrapper.decode, SerializeWrapper.encode(self.wrapped_fn))

    def forward(self, *runtime_args):
        if hasattr(self.wrapped_fn, "_boxed_call") and self.wrapped_fn._boxed_call:
            # Note: if the wrapped_fn is a boxed function, unbox it now
            args = []
            args.extend(runtime_args)
            return self.wrapped_fn(args)
        return self.wrapped_fn(*runtime_args)

    @staticmethod
    def encode(object):
        if isinstance(object, CompiledFxGraph):
            mod = copy.copy(object)
            mod.current_callable = None
            return (CompiledFxGraph, (mod,))
        elif isinstance(object, GraphModule):
            if len(object._modules) > 0:
                raise NotImplemented("Only fully-inlined graph module is supported")
            gm_dict = object.__dict__.copy()
            del gm_dict["_graph"]
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

    def decode(cls, arg_tuple):
        if cls == CompiledFxGraph:
            (compiled_fn,) = arg_tuple
            path = get_path(compiled_fn.cache_key, "py")[2]
            compiled_fn.current_callable = PyCodeCache.load_by_key_path(
                compiled_fn.cache_key,
                path,
                compiled_fn.cache_linemap,
                compiled_fn.constants,
            ).call
            return SerializeWrapper(compiled_fn)
        elif cls == GraphModule:
            gm_dict, graph_meta, nodes_meta = arg_tuple
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
            raise NotImplementedError(f"Unsupported decode: {cls}, {arg_tuple}")


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

    def save_gm(self, key, value: torch.fx.GraphModule, expire=None) -> torch.fx.GraphModule:
        # Note: since GraphModules ser/des may do canonicalization, so the cached version should be returned
        return value

    def load_gm(self, key):
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

    def save_gm(self, key, value: torch.fx.GraphModule, expire=None) -> torch.fx.GraphModule:
        artifact_path = self._graph_path(key)
        logger.info(f"Save cache in location: {artifact_path}")
        with compile_lock, _disable_current_modes():
            with open(artifact_path, "wb+") as f:
                pickle.dump(value, f)
            with open(artifact_path, "rb") as f:
                cached_graph = pickle.load(f)
        return cached_graph

    def load_gm(self, key):
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
        logger.debug(f"Use {cache_path} as default local cache")
    return XpuGraphLocalCache(cache_path)
