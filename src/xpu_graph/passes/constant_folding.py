import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from xpu_graph.config import OptLevel
from xpu_graph.constant_manager import get_constant_manager, is_constant
from xpu_graph.fx_utils import get_disable_fake_mode
from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.utils import logger

__all__ = ["ConstantFolding"]


def _no_folding(node: fx.Node):
    no_fold_call_function_list = [
        "air.scope_enter.default",
        "air.scope_exit.default",
    ]
    if node.op == "call_function":
        return str(node.target) in no_fold_call_function_list


class ConstantFolding(Optimizer):
    def __init__(self, folding_params=False):
        super().__init__()
        self.folding_params = folding_params

    def _all_input_constant(self, node: fx.Node):
        flat_args, _ = pytree.tree_flatten((node.args, node.kwargs))
        return all(is_constant(arg, self.folding_params) for arg in flat_args)

    def process(self, gm: torch.fx.GraphModule):
        changed = False
        graph = gm.graph
        get_attr_insert_point = None
        for get_attr_insert_point in gm.graph.nodes:
            if get_attr_insert_point.op != "get_attr" and get_attr_insert_point.op != "placeholder":
                break

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.is_impure() or _no_folding(node):
                continue

            if self._all_input_constant(node):
                changed = True

                def get_arg_value(arg):
                    if isinstance(arg, fx.Node):
                        return getattr(gm, arg.target)
                    return arg

                reconstructed_args, reconstructed_kwargs = pytree.tree_map(get_arg_value, (node.args, node.kwargs))

                logger.info(f"start constant folding: {node.name} {node.target}")

                disable_fake_mode = get_disable_fake_mode()
                with disable_fake_mode():
                    constant_value = node.target(*reconstructed_args, **reconstructed_kwargs)

                constant_name = get_constant_manager(gm).register_constant(constant_value, node.name)
                with graph.inserting_before(get_attr_insert_point):
                    constant_node = graph.create_node("get_attr", constant_name)
                    node.replace_all_uses_with(constant_node)

        return changed
