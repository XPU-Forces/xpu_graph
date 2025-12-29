import pytest
import torch

from xpu_graph.compiler import Target, XpuGraph, XpuGraphConfig


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.ops.aten._to_copy(self.linear(x).relu(), dtype=torch.int32)


class TestDeviceGraphCompiler:
    def setup_method(self):
        torch.mlu.set_device(0)
        self.module = MyModule().eval().mlu()

        self.device_graph_func = torch.compile(
            self.module,
            backend=XpuGraph(
                XpuGraphConfig(
                    False,
                    target=Target.mlu,
                    freeze=True,  # WARNING(liuyuan): Critical for nn.Module with Parameter under pytorch 2.5-
                    debug=True,
                    vendor_compiler_config={"compiler": "device_graph"},
                )
            ),
            dynamic=False,
            fullgraph=False,  # NOTE: allow graph breaks; Device Graph capture/replay may not support fullgraph for all models yet
        )
        assert self.device_graph_func is not None

    @pytest.mark.parametrize("shape", [(32,)])
    def testInference(self, shape):
        input = torch.randn((*shape, 4)).mlu()
        torch.testing.assert_close(
            self.module(input), self.device_graph_func(input), rtol=1e-03, atol=1e-03, equal_nan=True
        )
        input2 = torch.randn((*shape, 4)).mlu()
        torch.testing.assert_close(
            self.module(input2), self.device_graph_func(input2), rtol=1e-03, atol=1e-03, equal_nan=True
        )


if __name__ == "__main__":
    testObj = TestDeviceGraphCompiler()
    testObj.setup_method()
    testObj.testInference((32,))
