import numpy as np
import pytest
import torch
import torch._dynamo.config
import torch_npu
import triton
import triton.language as tl
from torch import fx, nn

import xpu_graph
from xpu_graph.accuracy_utils import assert_close, benchmark_compare_close


class NPU_Silumul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        d = input.shape[-1] // 2
        res = torch.nn.functional.silu(input[..., :d]) * input[..., d:]
        res1 = res * input[..., d:]
        return res1


def test_silu_mul():
    # create input data
    shape = [12, 37888]
    dtype = torch.bfloat16
    input = torch.randn(shape, dtype=dtype, device="npu")

    # new model
    model = NPU_Silumul().npu()
    res_eager = model(input)
    res_goden = model(input.to(torch.float32))

    # compile
    model_forward = model.forward
    from xpu_graph.compiler import OptLevel, Target, XpuGraph
    from xpu_graph.config import XpuGraphConfig

    xconf = XpuGraphConfig(
        is_training=False,
        debug=False,
        dump_graph=True,
        freeze=True,
        target=Target.npu,
        opt_level=OptLevel.level2,
        vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
    )
    compiled_model = torch.compile(model_forward, backend=XpuGraph(xconf), fullgraph=True)
    res_tri = compiled_model(input)

    # accuracy check
    try:
        assert_close(res_goden.to(torch.float32), res_tri.to(torch.float16))
    except Exception as e:
        print(e)
        print("starting benchmark compare_close:")
        benchmark_compare_close(res_goden.to(torch.float32), res_tri.to(torch.float16), res_eager.to(torch.float16))
        print("PASSED")


if __name__ == "__main__":
    test_silu_mul()
