import time

import torch
import torch.fx as fx
from abc import ABC, abstractmethod

from xpu_graph.config import OptLevel
from xpu_graph.utils import xpu_timer, logger

opt_times = 0


class Optimizer(ABC):
    _debug = False
    _dump_graph = False
    _opt_level = OptLevel.level0

    @abstractmethod
    def process(self, gm: fx.GraphModule) -> bool:
        pass

    # TODO(zhangjihang): Always close timer temporarily. Need a config to contral after.
    # @xpu_timer
    def __call__(self, gm: fx.GraphModule) -> bool:
        prev_nodes_num = len(gm.graph.nodes)

        changed = self.process(gm)

        if changed:
            logger.debug(
                f"{self.__class__.__bases__[0].__name__}.{self.__class__.__name__} changed graph, before: {prev_nodes_num} nodes, after: {len(gm.graph.nodes)} nodes."
            )

        if changed and self._dump_graph:
            global opt_times
            graph = fx.passes.graph_drawer.FxGraphDrawer(gm, self.__class__.__name__)
            graph.get_dot_graph().write_svg(
                f"{opt_times}_{self.__class__.__name__}.svg"
            )
            opt_times += 1

        return changed
