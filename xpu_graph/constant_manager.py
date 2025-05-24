import torch
import torch.fx as fx

constant_manager_map = {}


# Note: constants in traced GraphModule always startswith "_tensor_constant"
# See: torch/fx/_symbolic_trace.py : Tracer.create_arg()
_TRACED_CONST_PREFIX = "_tensor_constant"
_FOLDED_CONST_PREFIX = "_xpugraph_folded_"


class ConstantManager:
    def __init__(self, gm: fx.GraphModule):
        self._constant_id = 0
        self._gm = gm

    def register_constant(self, constant: torch.Tensor, name: str) -> str:
        constant_name = _FOLDED_CONST_PREFIX + name + f"_{self._constant_id}"
        # self._gm.register_buffer(constant_name, constant)
        self._gm.register_buffer(constant_name, constant)
        # setattr(self._gm, constant_name, torch.nn.Parameter(constant, requires_grad=False))
        self._constant_id += 1
        return constant_name


def get_constant_manager(gm):
    if gm not in constant_manager_map:
        constant_manager_map[gm] = ConstantManager(gm)

    return constant_manager_map[gm]


# TODO: Till now, only support get_attr node.
def is_constant(arg, freezing=False):
    if (
        isinstance(arg, fx.Node)
        and arg.op == "get_attr"
        and (
            freezing
            or arg.target.startswith(_TRACED_CONST_PREFIX)
            or arg.target.startswith(_FOLDED_CONST_PREFIX)
        )
    ):
        return True
    return False
