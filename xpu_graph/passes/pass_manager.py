import torch
import torch.fx as fx


class PassManager:
    def __init__(self, config):
        self.config = config

        from .optimizer import Optimizer

        Optimizer._debug = config.debug
        Optimizer._dump_graph = config.dump_graph

        self._passes = []
        from .patterns.pattern_manager import PatternManager

        self._pattern_manager = PatternManager(config)

        from .dce import Dce

        if Dce._opt_level <= config.opt_level:
            self._passes.append(Dce())

        from .cse import Cse

        if Cse._opt_level <= config.opt_level:
            self._passes.append(Cse())

        if config.constant_folding:
            from .constant_folding import ConstantFolding

            self._passes.append(ConstantFolding())

        self._passes.append(self._pattern_manager)

    def __call__(self, gm: fx.GraphModule, example_inputs):
        changed = True
        while changed:
            from torch.fx.passes.shape_prop import ShapeProp

            ShapeProp(gm).propagate(*example_inputs)
            changed = False
            for pass_ in self._passes:
                changed = changed or pass_(gm)

        gm.recompile()

        if self.config.enable_cache:
            # Note: Currently, we only inline modules with a E2E make_fx, just for serialize / desrialize
            from torch.fx.experimental.proxy_tensor import make_fx

            gm = make_fx(gm, record_module_stack=True)(*example_inputs)

        return gm

    def get_pattern_manager(self):
        return self._pattern_manager
