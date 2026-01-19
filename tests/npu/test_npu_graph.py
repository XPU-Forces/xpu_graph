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

    def test_npu_graph_runner_stream_sync(self):
        H = 8192
        L = 64
        model = torch.nn.Sequential(*(torch.nn.Linear(H, H) for _ in range(L))).npu()
        input_tensor = torch.empty(1024, H, device="npu").uniform_(10, 100)

        golden = model(input_tensor)
        golden = golden + 2.0

        device_graph = GraphRunner[Target.npu](
            model,
            lambda input_tensor: input_tensor,
            lambda input_buffer, input_tensor: (input_buffer.copy_(input_tensor), True),
        )

        device_graph.capture(torch.empty_like(input_tensor).uniform_(1, 2))

        replay_stream = torch.npu.Stream()
        with replay_stream:
            result = device_graph.forward(input_tensor)

        s = torch.npu.Stream()
        s.wait_stream(replay_stream)
        with torch.npu.stream(s):
            result = result + 2.0

        torch.npu.synchronize()

        assert torch.allclose(result, golden)
