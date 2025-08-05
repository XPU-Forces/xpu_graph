from typing import Set

import torch
import torch.fx as fx

from xpu_graph.constant_manager import _ALL_CONSTANT_PREFIXES
from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.utils import logger

__all__ = ["ConstantPruning"]


class ConstantPruning(Optimizer):
    def __init__(self, include_params=False):
        super().__init__()
        self.include_params = include_params

    def _collect_alive_constants(self, graph: fx.Graph) -> Set[str]:
        alive_constants = set()
        for node in graph.nodes:
            if node.op == "get_attr":
                if node.target.startswith(_ALL_CONSTANT_PREFIXES):
                    alive_constants.add(node.target)
        return alive_constants

    def process(self, gm: torch.fx.GraphModule) -> bool:
        changed = False

        alive_constants = self._collect_alive_constants(gm.graph)
        logger.debug(f"[ConstantPruning] Found {len(alive_constants)} alive constants in the graph: {alive_constants}")

        all_attribute_names = list(gm.__dict__.keys())

        for name in all_attribute_names:
            if name.startswith(_ALL_CONSTANT_PREFIXES):
                if name not in alive_constants:
                    try:
                        delattr(gm, name)

                        if name in gm._buffers:
                            del gm._buffers[name]
                        if name in gm._parameters:
                            del gm._parameters[name]

                        changed = True
                        logger.info(f"[ConstantPruning] Pruned unused constant attribute: {name}")
                    except AttributeError:
                        logger.warning(f"[ConstantPruning] Tried to delete {name}, but it was not found.")

        return changed
