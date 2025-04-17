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
            self.__dump_files(gm)
        return changed
    
    def __dump_files(self, gm):
        import os
        import shutil
        from subprocess import run

        global opt_times
        dirname = "xpu_graph_debugs"

        if opt_times == 0:
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            if run("dot -h", shell=True).returncode != 0:
                # TODO(liuyuan): Maybe use yum according to the kernel.
                # NOTE(liuyuan): graphviz should be installed for the following method [get_dot_graph()]
                run("apt install -y graphviz", shell=True, check=True)

        filename = os.path.join(dirname, f"optimization_{opt_times}_after_pass_{self.__class__.__name__}")
        
        # NOTE(liuyuan): write irs into file.
        with open(f"{filename}.ll", 'w') as f:
            f.writelines(str(gm.graph))

        # NOTE(liuyuan): visualize the graph and dump as svg file.
        graph = fx.passes.graph_drawer.FxGraphDrawer(gm, self.__class__.__name__)
        graph.get_dot_graph().write_svg(
            f"{filename}.svg"
        )
        opt_times += 1
