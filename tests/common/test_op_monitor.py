import torch

aten = torch.ops.aten

from xpu_graph.monitors import OpMonitor
from xpu_graph.test_utils import need_xpu_graph_logs


def test_op_monitor(caplog):
    def slow_add_tensor(a: torch.Tensor, b: torch.Tensor):
        a_np = a.numpy(force=True)
        b_np = b.numpy(force=True)
        c_np = a_np + b_np
        return torch.from_numpy(c_np)

    with need_xpu_graph_logs():
        with OpMonitor({aten.add.Tensor: slow_add_tensor}):
            x = torch.randn(2, 2)
            y = torch.randn(2, 2)
            c = x + y
            print(c)

    assert f"Monitored op: {aten.add.Tensor}" in caplog.text and "diverges" not in caplog.text


def test_op_monitor_fail(caplog):
    with need_xpu_graph_logs():
        with OpMonitor({aten.add.Tensor: aten.sub.Tensor}):
            x = torch.randn(2, 2)
            y = torch.randn(2, 2)
            c = x + y
            print(c)

    assert f"Monitored op: {aten.add.Tensor}" in caplog.text and "diverges" in caplog.text
