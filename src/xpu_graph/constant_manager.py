import hashlib
from collections import defaultdict

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

        import io

        # TODO(liuyuan): evaluta the performance.
        io_buffer = io.BytesIO()
        torch.save(tensor.detach(), io_buffer)
        tensor_bytes = io_buffer.getvalue()

        content_hash = hashlib.sha256(tensor_bytes).hexdigest()

        return f"{meta_hash}-{content_hash}"


class ConstantManager:
    def __init__(self, gm: fx.GraphModule):
        self._gm = gm
        self._constant_id = 0
        self._hash_to_name_map = {}
        self._initialize_from_graph()

    def _initialize_from_graph(self):
        temp_hash_to_names = defaultdict(list)
        all_constants = {**dict(self._gm.named_buffers()), **dict(self._gm.named_parameters())}

        for name, const_tensor in all_constants.items():
            if name.startswith(_ALL_CONSTANT_PREFIXES):
                tensor_hash = _get_tensor_hash(const_tensor)
                temp_hash_to_names[tensor_hash].append(name)

        for tensor_hash, names_list in temp_hash_to_names.items():
            if len(names_list) > 1:
                primary_name = names_list[0]
                logger.info(
                    f"[ConstantManager] Found redundant constants for hash {tensor_hash[:8]}... Using '{primary_name}' as primary."
                )

                for redundant_name in names_list[1:]:
                    for node in self._gm.graph.nodes:
                        if node.op == "get_attr" and node.target == redundant_name:
                            logger.debug(f"Redirecting get_attr '{redundant_name}' to '{primary_name}'")
                            node.target = primary_name

            self._hash_to_name_map[tensor_hash] = names_list[0]

    def _print_constants(self):
        all_constants = []
        for name, _ in self._gm.named_parameters():
            all_constants.append(name)
        for name, _ in self._gm.named_buffers():
            all_constants.append(name)

        if all_constants:
            for const_name in sorted(all_constants):
                logger.debug(f"Found {len(all_constants)} managed constants in the GraphModule: {const_name}")
        else:
            logger.debug(f"No managed constants found.")

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

    def remove_redundant_constants(self) -> bool:
        changed = False

        alive_constants = {
            node.target
            for node in self._gm.graph.nodes
            if node.op == "get_attr" and node.target.startswith(_ALL_CONSTANT_PREFIXES)
        }

        all_attribute_names = set(dict(self._gm.named_parameters(recurse=False)).keys()).union(
            dict(self._gm.named_buffers(recurse=False)).keys(), self._gm.__dict__.keys()
        )

        for name in all_attribute_names:
            if name.startswith(_ALL_CONSTANT_PREFIXES):
                if name not in alive_constants:
                    try:
                        if hasattr(self._gm, name):
                            delattr(self._gm, name)

                            logger.debug(f"Removed unused constant: {name}")
                            changed = True
                    except AttributeError:
                        logger.warning(f"Tried to delete {name}, but failed.")

        if changed:
            hashes_to_delete = [h for h, n in self._hash_to_name_map.items() if n not in alive_constants]
            for h in hashes_to_delete:
                del self._hash_to_name_map[h]

        self._print_constants()
        return changed


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
