import hashlib
from functools import lru_cache

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from packaging import version
from torch import SymInt

from .fx_utils import get_disable_fake_mode
from .utils import logger

constant_manager_map = {}


# Note: constants in traced GraphModule always startswith "_tensor_constant"
# See: torch/fx/_symbolic_trace.py : Tracer.create_arg()
_TRACED_CONST_PREFIX = "_tensor_constant"
# Prefixes for constants lifted from nn.Module attributes by Dynamo
_DYNAMO_FN_PARAM_PREFIX = "fn____parameters__"
_DYNAMO_FN_BUFFER_PREFIX = "fn____buffers__"
_DYNAMO_PARAM_PREFIX = "self____parameters__"
_DYNAMO_BUFFER_PREFIX = "self____buffers__"
# Our own prefix for folded constants
_FOLDED_CONST_PREFIX = "_xpugraph_folded_"

_ALL_CONSTANT_PREFIXES = (
    _TRACED_CONST_PREFIX,
    _DYNAMO_PARAM_PREFIX,
    _DYNAMO_BUFFER_PREFIX,
    _DYNAMO_FN_PARAM_PREFIX,
    _DYNAMO_FN_BUFFER_PREFIX,
    _FOLDED_CONST_PREFIX,
)


def _get_tensor_hash(tensor: torch.Tensor) -> str:
    # We include shape, dtype, device, and strides to ensure that two tensors are
    # only considered identical if they are logically and physically equivalent.
    disable_fake_mode = get_disable_fake_mode()
    with disable_fake_mode():
        meta = (
            tensor.shape,
            tensor.dtype,
            # device object is not hashable, so we convert it to a string.
            str(tensor.device),
            tensor.stride(),
        )
        meta_hash = hashlib.sha256(str(meta).encode()).hexdigest()

        tensor_bytes = tensor.cpu().numpy().tobytes()

        content_hash = hashlib.sha256(tensor_bytes).hexdigest()

        return f"{meta_hash}-{content_hash}"


class ConstantManager:
    def __init__(self, gm: fx.GraphModule):
        self._gm = gm
        self._constant_id = 0

        self._hash_to_name_map = {}

        self._initialize_from_graph()

    def _initialize_from_graph(self):
        all_constants = {**dict(self._gm.named_buffers()), **dict(self._gm.named_parameters())}

        for name, const_tensor in all_constants.items():
            if name.startswith(_ALL_CONSTANT_PREFIXES):
                tensor_hash = _get_tensor_hash(const_tensor)
                if tensor_hash not in self._hash_to_name_map:
                    self._hash_to_name_map[tensor_hash] = name

    def register_constant(self, constant: torch.Tensor, name: str) -> str:
        constant_hash = _get_tensor_hash(constant)

        if constant_hash in self._hash_to_name_map:
            return self._hash_to_name_map[constant_hash]
        else:
            constant_name = _FOLDED_CONST_PREFIX + name + f"_{self._constant_id}"

            self._gm.register_buffer(constant_name, constant)
            self._constant_id += 1

            self._hash_to_name_map[constant_hash] = constant_name

            return constant_name


def get_constant_manager(gm):
    if gm not in constant_manager_map:
        constant_manager_map[gm] = ConstantManager(gm)

    return constant_manager_map[gm]


def is_constant(arg, include_params=False):
    if not isinstance(arg, fx.Node):
        leaves = pytree.tree_leaves(arg)
        return not any(isinstance(leaf, SymInt) for leaf in leaves)

    if arg.op != "get_attr":
        return False

    if not arg.target.startswith(_ALL_CONSTANT_PREFIXES):
        return False

    if not include_params and (
        arg.target.startswith(_DYNAMO_PARAM_PREFIX) or arg.target.startswith(_DYNAMO_BUFFER_PREFIX)
    ):
        return False

    return True
