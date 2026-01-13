import torch

from xpu_graph.config import Target
from xpu_graph.device_graph_runner import GraphRunner


class TestNpuGraphRunner:
    def test_npu_graph_runner_basic(self):
        model = torch.nn.Sequential(*[torch.nn.Linear(1024, 1024)] * 3).npu()
        input_tensor = torch.empty(1024, 1024).uniform_(10, 100).npu()

        golden = model(input_tensor)
        device_graph = GraphRunner[Target.npu](
            model,
            lambda input_tensor: input_tensor,
            lambda input_buffer, input_tensor: (input_buffer.copy_(input_tensor), True),
        )
        device_graph.capture(torch.empty_like(input_tensor).uniform_(1, 2))

        assert torch.allclose(device_graph(input_tensor), golden)

        device_graph_2 = device_graph.clone()
        device_graph_2.capture(torch.empty_like(input_tensor).uniform_(3, 4), clone_args=True)

        assert torch.allclose(device_graph_2(input_tensor), golden)
