import pytest
import torch

from xpu_graph import Target
from xpu_graph.compiler import XpuGraphCompiler


class TestNpuCompilationCache:
    def setup_method(self):
        self.compiler = XpuGraphCompiler()
        self.compiler.set_target(Target.npu).set_vendor_compiler_config(
            {"compiler": "ge", "mode": "reduce-overhead"}
        ).set_debug(True).set_is_training(False).set_enable_cache(True).done()

    def test_basice(self):
        def dummy(x, y):
            return x @ y

        inputs = (torch.empty(1024), torch.empty(1024))
        compiled = self.compiler.compile(dummy, dynamic=False)
        compiled(*inputs)
