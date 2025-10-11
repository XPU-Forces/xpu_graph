from pathlib import Path

import pytest
import torch

from xpu_graph.super_kernel_launcher import xpu_graph_super_kernel


@pytest.mark.datafiles(Path(__file__).parent / "test_files/sk_add.o")
def test_super_kernel_launcher_basic(datafiles):
    inputs = [torch.empty(40, 2048, 120).to(torch.int16).npu() for _ in range(6)]
    outputs = [torch.empty(40, 2048, 120).to(torch.int16).npu() for _ in range(3)]
    s = torch.npu.Stream()

    xpu_graph_super_kernel(inputs, outputs, s.npu_stream, str(datafiles / "sk_add.o"), "sk_add")
    s.synchronize()
