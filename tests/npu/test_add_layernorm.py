import numpy as np
import pytest
import torch
import torch._dynamo.config
import torch_npu
import triton
import triton.language as tl
from torch import fx, nn

import xpu_graph
from xpu_graph import OptLevel, Target, XpuGraph, XpuGraphConfig
from xpu_graph.accuracy_utils import assert_close, benchmark_compare_close
from xpu_graph.test_utils import need_xpu_graph_logs


def call_aten_kernel(input, residual, weight, bias):
    new_residual_tensor = input.to(torch.float32) + residual.to(torch.float32)
    new_residual = new_residual_tensor.to(torch.bfloat16)
    eps = 1e-6
    x = new_residual
    mean = x.mean(-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(variance + eps)
    y_ref = weight * x_norm + bias
    y_ref = y_ref.to(input.dtype)
    return y_ref


class NPU_LayerNormWithResidual(nn.Module):
    def __init__(self, weight, bias, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, input, residual, arg486_1, arg487_1, weight=None, bias=None):
        """
        1. 将输入和残差转换为float32并相加
        2. 转回bfloat16精度
        3. 应用NPU LayerNorm
        """
        quant_matmul_v2_111 = input
        npu_dtype_cast_164 = residual
        arg485_1 = weight
        arg485_2 = bias
        npu_dtype_cast_165 = torch.ops.npu.npu_dtype_cast.default(quant_matmul_v2_111, torch.float32)
        quant_matmul_v2_111 = None
        npu_dtype_cast_166 = torch.ops.npu.npu_dtype_cast.default(npu_dtype_cast_164, torch.float32)
        npu_dtype_cast_164 = None
        add_55 = torch.ops.aten.add.Tensor(npu_dtype_cast_165, npu_dtype_cast_166)
        npu_dtype_cast_165 = npu_dtype_cast_166 = None
        npu_dtype_cast_167 = torch.ops.npu.npu_dtype_cast.default(add_55, torch.bfloat16)
        add_55 = None
        # 使用layer_norm替代rms_norm
        layer_norm_56 = torch.nn.functional.layer_norm(
            npu_dtype_cast_167, 
            npu_dtype_cast_167.shape[-1:], 
            weight=arg485_1, 
            bias=arg485_2, 
            eps=self.eps
        )
        npu_dtype_cast_167 = arg485_1 = arg485_2 = None
        return layer_norm_56


def compare_add_layernorm_pattern(xpu_graph_backend):
    # create input data
    shape = [1, 3584]
    dtype = torch.bfloat16
    input = torch.randn(shape, dtype=dtype).npu()
    input = input.clamp(0, 10)
    residual = torch.randn(shape, dtype=dtype).npu()
    weight = torch.randn(shape, dtype=dtype).npu()
    bias = torch.randn(shape, dtype=dtype).npu()
    arg486_1 = torch.tensor([0], dtype=torch.int32).npu()
    arg487_1 = torch.randn(size=(152064, 3584), dtype=torch.bfloat16).npu()

    # init our graph
    model = NPU_LayerNormWithResidual(weight, bias).npu()
    model_forward = model.forward
    compiled_model = torch.compile(model_forward, backend=xpu_graph_backend, dynamic=False)

    # get result
    mm_out = compiled_model(input, residual, arg486_1, arg487_1, weight, bias)

    mm_ref_fp32 = model(
        input.to(torch.float32),
        residual.to(torch.float32),
        arg486_1,
        arg487_1.to(torch.float32),
        weight.to(torch.float32),
        bias.to(torch.float32),
    )
    mm_ref_bf16 = call_aten_kernel(input, residual, weight, bias)

    try:
        assert_close(mm_ref_fp32.to(torch.float32), mm_out)
    except Exception as e:
        print(e)
        print("starting benchmark compare_close:")
        benchmark_compare_close(mm_ref_fp32.to(torch.float32), mm_out, mm_ref_bf16)
        print("PASSED")


class TestAddLayerNormPattern:
    def setup_class(self):
        config = XpuGraphConfig(
            is_training=False,
            dump_graph=True,
            freeze=True,
            target=Target.npu,
            opt_level=OptLevel.level2,
            vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
            debug=False,
        )
        self.xpu_graph_backend = XpuGraph(config)

    def test_add_layernorm_pattern(self, caplog):
        with need_xpu_graph_logs():
            compare_add_layernorm_pattern(self.xpu_graph_backend)
        # 注意：这里需要检查FusedAddLayernorm是否存在，如果不存在则检查FusedLayerNorm
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text


if __name__ == "__main__":
    config = XpuGraphConfig(
        is_training=False,
        dump_graph=True,
        freeze=True,
        target=Target.npu,
        opt_level=OptLevel.level2,
        vendor_compiler_config={"mode": "reduce-overhead", "compiler": "ge"},
        debug=True,
    )
    xpu_graph_backend = XpuGraph(config)
    compare_add_layernorm_pattern(xpu_graph_backend)