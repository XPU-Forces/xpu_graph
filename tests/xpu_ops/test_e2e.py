import torch
import torch.nn as nn
import torch_npu
import xpu_ops
from torch.nn import Parameter

import xpu_graph
import xpu_graph.config

xpu_ops.load_xpu_ops_npu()

M = 1
K = 7392
N = 8192

layer_num = 1


def test_e2e():
    class AscendW8A8Linear(nn.Module):
        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            out_dtype=torch.bfloat16,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = torch.randn(out_features, in_features).to(dtype=torch.int8).npu()

            self.in_scale = Parameter(torch.ones(in_features, dtype=torch.bfloat16).npu() * 0.5)
            self.out_scale = Parameter(torch.randn(out_features, dtype=torch.bfloat16).npu())
            self.bias = Parameter(torch.randn(out_features, dtype=torch.bfloat16).npu())
            self.out_dtype = out_dtype

        def forward(self, x):
            for _ in range(layer_num):
                x = x / self.in_scale
                x, x_scale = torch_npu.npu_dynamic_quant(x)
                x_scale = x_scale.view(-1)
                out_shape = x.size()[:-1] + (self.out_features,)
                x = x.view(-1, x.size(-1))
                x = torch_npu.npu_quant_matmul(
                    x,
                    self.weight.t(),
                    self.out_scale,
                    pertoken_scale=x_scale,
                    bias=self.bias,
                    output_dtype=self.out_dtype,
                )
                x = x.view(out_shape)
            return x

    model = AscendW8A8Linear(K, N)
    x = torch.ones(M, K).to(dtype=torch.bfloat16).npu()
    expect = model(x)

    from xpu_graph.compiler import XpuGraph
    from xpu_graph.config import XpuGraphConfig

    config = XpuGraphConfig()
    config.target = xpu_graph.config.Target.npu
    config.vendor_compiler_config = {"compiler": "ge"}
    config.debug = True
    config.opt_level = xpu_graph.config.OptLevel.level2
    config.execute_mode = xpu_graph.config.ExecuteMode.graph
    compiled_model = torch.compile(model, backend=XpuGraph(config))

    res = compiled_model(x)

    from xpu_graph.test_utils import is_similar

    assert is_similar(expect, res)
