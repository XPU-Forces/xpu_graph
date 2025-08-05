import hashlib

import torch
import torch.fx as fx
from packaging import version

disable_fake_mode = None
torch_version = version.parse(torch.__version__.split("+")[0])  # Handle dev versions like '2.5.0.dev20240000'
if torch_version < version.parse("2.5"):
    from torch.fx.experimental.proxy_tensor import (
        maybe_disable_fake_tensor_mode as disable_fake_mode,
    )
else:
    from torch._subclasses.fake_tensor import (
        unset_fake_temporarily as disable_fake_mode,
    )

constant_manager_map = {}


# Note: constants in traced GraphModule always startswith "_tensor_constant"
# See: torch/fx/_symbolic_trace.py : Tracer.create_arg()
_TRACED_CONST_PREFIX = "_tensor_constant"
# Prefixes for constants lifted from nn.Module attributes by Dynamo
_DYNAMO_PARAM_PREFIX = "self____parameters__"
_DYNAMO_BUFFER_PREFIX = "self____buffers__"
# Our own prefix for folded constants
_FOLDED_CONST_PREFIX = "_xpugraph_folded_"

_ALL_CONSTANT_PREFIXES = (
    _TRACED_CONST_PREFIX,
    _DYNAMO_PARAM_PREFIX,
    _DYNAMO_BUFFER_PREFIX,
    _FOLDED_CONST_PREFIX,
)


def _get_tensor_hash(tensor: torch.Tensor) -> str:
    # We include shape, dtype, device, and strides to ensure that two tensors are
    # only considered identical if they are logically and physically equivalent.
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
        # Backup plan
        # try:
        #     tensor_bytes = tensor.cpu().numpy().tobytes()
        # except Exception as e:
        #     tensor_bytes = str(tensor).encode()

        content_hash = hashlib.sha256(tensor_bytes).hexdigest()

        return f"{meta_hash}-{content_hash}"


class ConstantManager:
    def __init__(self, gm: fx.GraphModule):
        self._gm = gm
        self._constant_id = 0

        self._hash_to_name_map = {}

        self._managed_constants = set()
        self._initialize_from_graph()

    def _initialize_from_graph(self):
        all_constants = {**dict(self._gm.named_buffers()), **dict(self._gm.named_parameters())}

        for name, const_tensor in all_constants.items():
            if name.startswith(_ALL_CONSTANT_PREFIXES):
                tensor_hash = _get_tensor_hash(const_tensor)
                if tensor_hash not in self._hash_to_name_map:
                    self._hash_to_name_map[tensor_hash] = name
                self._managed_constants.add(name)

    def register_constant(self, constant: torch.Tensor, name: str) -> str:
        constant_hash = _get_tensor_hash(constant)

        if constant_hash in self._hash_to_name_map:
            return self._hash_to_name_map[constant_hash]
        else:
            constant_name = _FOLDED_CONST_PREFIX + name + f"_{self._constant_id}"

            self._gm.register_buffer(constant_name, constant)
            self._constant_id += 1

            self._hash_to_name_map[constant_hash] = constant_name
            self._managed_constants.add(constant_name)

            return constant_name


def get_constant_manager(gm):
    if gm not in constant_manager_map:
        constant_manager_map[gm] = ConstantManager(gm)

    return constant_manager_map[gm]


# TODO: Till now, only support get_attr node.
def is_constant(arg, include_params=False):
    if isinstance(arg, fx.Node) and arg.op == "get_attr":
        if arg.target.startswith(_ALL_CONSTANT_PREFIXES):
            if not include_params and (
                arg.target.startswith(_DYNAMO_PARAM_PREFIX) or arg.target.startswith(_DYNAMO_BUFFER_PREFIX)
            ):
                return False
            return True
    return False
