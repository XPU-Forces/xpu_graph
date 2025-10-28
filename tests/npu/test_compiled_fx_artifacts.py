import os
import pickle
import pytest

import torch
import torch_npu
import torchair

from xpu_graph.compiler import Target, XpuGraph, XpuGraphConfig


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        z = x + y
        return z


class TestCompiledFxArtifacts:
    def setup_method(self):
        torch.npu.set_device(0)

    def test_dump_and_load_artifacts(self):
        config = XpuGraphConfig(
            False,
            target=Target.npu,
            freeze=True,
            debug=True,
            vendor_compiler_config={"compiler": "ge", "mode": "reduce-overhead"},
        )
        xpu_backend = XpuGraph(config)
        """
        The NPU backend (xpu_graph/backends/npu.py) needs to add the compiler call
        in the definition of ge_compiler before the function return:
        ...
        from torchair.npx_fx_compiler import CompiledFxGraph, _CompiledFxArtifacts
        
        ...
        compiler = tng.get_compiler(compiler_config=config)
        compiled_fx: _CompiledFxGraph = compiler(module, example_inputs)
        dump_artifacts: _CompiledFxArtifacts = compiled_fx.dump_artifacts()
        
        with open('test_artifacts.pkl', 'wb') as file:
            pickle.dump(dump_artifacts, file)
        
        ...
        """
        compiled_model = torch.compile(Model.npu(), backend=xpu_backend, dynamic=False, fullgraph=True)

        x = torch.tensor([1.0]).npu()
        y = torch.tensor([2.0]).npu()

        z = compiled_model(x, y)

        compiled_model_from_artifacts = None
        with open('test_artifacts.pkl', 'wb') as file:
            artifacts = pickle.load(file)
            compiled_model_from_artifacts = torchair.npx_fx_compiler._CompiledFxGraph.load_artifacts(artifacts)

        assert compiled_model_from_artifacts != None

        another_z = compiled_model_from_artifacts(x, y)

        assert another_z == z


if __name__ == "__main__":
    testObj = TestCompiledFxArtifacts()
    testObj.setup_method()
    testObj.test_dump_and_load_artifacts()