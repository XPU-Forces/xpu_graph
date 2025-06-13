import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from xpu_graph.config import OptLevel
from xpu_graph.constant_manager import get_constant_manager, is_constant
from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.utils import logger

__all__ = ["ConstantFolding"]


def _no_folding(node: fx.Node):
    no_fold_call_function_list = []
    if node.op == "call_function":
        return node.target in no_fold_call_function_list


class ConstantFolding(Optimizer):
    def __init__(self, folding_params=False):
        super().__init__()
        self.folding_params = folding_params

    def _all_input_constant(self, node: fx.Node):
        return all(is_constant(arg, self.folding_params) for arg in node.args)

    def process(self, gm: torch.fx.GraphModule):
        changed = False
        graph = gm.graph
        get_attr_insert_point = None

        # For better readability, we insert get_attr node in the front of graph
        for get_attr_insert_point in gm.graph.nodes:
            if get_attr_insert_point.op != "get_attr" and get_attr_insert_point.op != "placeholder":
                break

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if _no_folding(node):
                continue
            if self._all_input_constant(node):
                changed = True
                new_args = (getattr(gm, arg.target) for arg in node.args)

                disable_fake_mode = None
                from packaging import version

                torch_version = version.parse(torch.__version__[:5])
                if torch_version < version.parse("2.5"):
                    from torch.fx.experimental.proxy_tensor import (
                        maybe_disable_fake_tensor_mode as disable_fake_mode,
                    )
                else:
                    from torch._subclasses.fake_tensor import (
                        unset_fake_temporarily as disable_fake_mode,
                    )

                logger.info(f"start constant folding: {node.name} {node.target}")

                with disable_fake_mode():
                    constant_value = node.target(*new_args, **node.kwargs)

                constant_name = get_constant_manager(gm).register_constant(constant_value, node.name)
                with graph.inserting_before(get_attr_insert_point):
                    constant_node = graph.create_node("get_attr", constant_name)
                    node.replace_all_uses_with(constant_node)

                logger.debug(f"constant folding graph: {graph}")

        gm.graph.lint()

        return changed
