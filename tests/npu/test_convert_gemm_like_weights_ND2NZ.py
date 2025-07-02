import pytest
import torch
import torch_npu

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    is_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "npu"
data_type = torch.float16


class NpuQuantMatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x2_data = torch.randint(0, 29, (16, 16), dtype=torch.int8, device=device)
        self.register_buffer("x2_data", x2_data, persistent=True)

    def forward(self, x1, scale):
        return torch.ops.npu.npu_quant_matmul.default(x1, self.x2_data, scale)


class NpuWeightQuantBatchmatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        M, K, N = 256, 128, 64
        weight = torch.randint(low=-8, high=8, size=(K, N), dtype=torch.int8, device=device)
        antiquant_scale = torch.randn((1, N), dtype=torch.bfloat16, device=device)
        antiquant_offset = torch.randn((1, N), dtype=torch.bfloat16, device=device)

        self.register_buffer("weight", weight, persistent=True)
        self.register_buffer("antiquant_scale", antiquant_scale, persistent=True)
        self.register_buffer("antiquant_offset", antiquant_offset, persistent=True)

    def forward(self, x):
        return torch_npu.npu_weight_quant_batchmatmul(
            x,
            self.weight,
            self.antiquant_scale,
            self.antiquant_offset,
            None,
            None,
            None,
            0,  # quant_scale, quant_offset, bias, antiquant_group_size
        )


class NpuMlaPrologModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        He, Hcq, Hckv = 7168, 1536, 512
        N, D, Dr = 32, 128, 64

        w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16, device=device)
        w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=device)
        w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16, device=device)

        self.register_buffer("w_dq", w_dq, persistent=True)
        self.register_buffer("w_uq_qr", w_uq_qr, persistent=True)
        self.register_buffer("w_dkv_kr", w_dkv_kr, persistent=True)

        self.rmsnorm_epsilon_cq = 1e-5
        self.rmsnorm_epsilon_ckv = 1e-5
        self.cache_mode = "PA_BSND"

    def forward(
        self, token_x, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, w_uk
    ):
        return torch_npu.npu_mla_prolog(
            token_x,
            self.w_dq,
            self.w_uq_qr,
            w_uk,
            self.w_dkv_kr,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            rope_sin,
            rope_cos,
            cache_index,
            kv_cache,
            kr_cache,
            rmsnorm_epsilon_cq=self.rmsnorm_epsilon_cq,
            rmsnorm_epsilon_ckv=self.rmsnorm_epsilon_ckv,
            cache_mode=self.cache_mode,
        )


class NpuConvertWeightToInt4packModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight_shape = (128, 128)  # 8-aligned dimensions
        weight = torch.randint(-8, 8, weight_shape, dtype=torch.int32, device=device)
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, inner_k_tiles):
        return torch.ops.npu.npu_convert_weight_to_int4pack.default(self.weight, inner_k_tiles)


class NpuGroupedMatmulFinalizeRoutingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        E, H, D = 8, 768, 768  # num_experts, hidden_dim, expert_dim
        w = torch.randn(E, H, D, dtype=torch.bfloat16, device=device)
        self.register_buffer("w", w, persistent=True)

    def forward(self, x, group_list):
        return torch_npu.npu_grouped_matmul_finalize_routing(
            x,
            self.w,
            group_list,
            scale=None,
            bias=None,
            pertoken_scale=None,
            shared_input=None,
            logit=None,
            row_index=None,
            dtype=None,
            shared_input_weight=1.0,
            shared_input_offset=0,
            output_bs=0,
            group_list_type=1,
        )


class FormatConversionAvoidanceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x2_data = torch.randint(0, 29, (8, 8), dtype=torch.int8, device=device)
        self.register_buffer("x2_data", x2_data, persistent=True)

    def forward(self, x1, scale):
        result1 = torch.ops.npu.npu_quant_matmul.default(x1, self.x2_data, scale)
        result2 = torch.ops.npu.npu_quant_matmul.default(result1, self.x2_data, scale)
        return result2


class TorchMmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight1 = torch.randn(128, 64, dtype=data_type, device=device)
        weight2 = torch.randn(64, 32, dtype=data_type, device=device)
        self.register_buffer("weight1", weight1, persistent=True)
        self.register_buffer("weight2", weight2, persistent=True)

    def forward(self, x):
        # First mm operation: x (batch, 128) @ weight1 (128, 64) -> (batch, 64)
        intermediate = torch.mm(x, self.weight1)
        # Second mm operation: intermediate (batch, 64) @ weight2 (64, 32) -> (batch, 32)
        result = torch.mm(intermediate, self.weight2)
        return result


class TorchMmBothConstantsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        matrix_a = torch.randn(64, 128, dtype=data_type, device=device)
        matrix_b = torch.randn(128, 32, dtype=data_type, device=device)
        self.register_buffer("matrix_a", matrix_a, persistent=True)
        self.register_buffer("matrix_b", matrix_b, persistent=True)

    def forward(self):
        return torch.mm(self.matrix_a, self.matrix_b)


def create_test_data(func_name):
    if func_name == "npu_quant_matmul":
        x1 = torch.randint(0, 8, (16, 16), dtype=torch.int8, device=device)
        scale = torch.randn(16, dtype=torch.float32, device=device)
        module = NpuQuantMatmulModule().to(device)
        return module, (x1, scale)

    elif func_name == "npu_weight_quant_batchmatmul":
        torch.manual_seed(42)
        M, K, N = 256, 128, 64
        x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
        module = NpuWeightQuantBatchmatmulModule().to(device)
        return module, (x,)

    elif func_name == "npu_mla_prolog":
        B, He, Hcq, Hckv = 2, 7168, 1536, 512
        N, D, Dr = 32, 128, 64
        S, Nkv = 2, 1
        BlockSize, BlockNum = 128, 64

        token_x = torch.rand(B, S, He, dtype=torch.bfloat16, device=device)
        w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16, device=device)
        rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, device=device)
        rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, device=device)
        rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16, device=device)
        rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16, device=device)
        cache_index = torch.randint(0, BlockNum, (B, S), dtype=torch.int64, device=device)
        kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16, device=device)
        kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16, device=device)

        module = NpuMlaPrologModule().to(device)
        return module, (
            token_x,
            rope_sin,
            rope_cos,
            cache_index,
            kv_cache,
            kr_cache,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            w_uk,
        )

    elif func_name == "npu_convert_weight_to_int4pack":
        inner_k_tiles = 0
        module = NpuConvertWeightToInt4packModule().to(device)
        return module, (inner_k_tiles,)

    elif func_name == "npu_grouped_matmul_finalize_routing":
        torch.manual_seed(42)
        B, S, H = 4, 32, 768  # batch, sequence, hidden

        x = torch.randn(B * S, H, dtype=torch.bfloat16, device=device)

        group_list = torch.tensor(
            [
                [0, 32, 0],  # tokens 0-31 to expert 0
                [32, 64, 1],  # tokens 32-63 to expert 1
                [64, 96, 2],  # tokens 64-95 to expert 2
                [96, 128, 3],  # tokens 96-127 to expert 3
            ],
            dtype=torch.int64,
            device=device,
        )

        module = NpuGroupedMatmulFinalizeRoutingModule().to(device)
        return module, (x, group_list)

    elif func_name == "format_conversion_avoidance":
        x1 = torch.randint(0, 8, (8, 8), dtype=torch.int8, device=device)
        scale = torch.randn(8, dtype=torch.float32, device=device)
        module = FormatConversionAvoidanceModule().to(device)
        return module, (x1, scale)

    elif func_name == "torch_mm":
        x = torch.randn(16, 128, dtype=data_type, device=device)
        module = TorchMmModule().to(device)
        return module, (x,)

    else:
        raise ValueError(f"Unknown function name: {func_name}")


def run_nd_to_nz_test(xpu_graph_backend, func_name, expected_pattern=None):
    module, input_args = create_test_data(func_name)

    with torch.no_grad():
        module.eval()
        result_direct = module(*input_args)

        compiled_module = torch.compile(module, backend=xpu_graph_backend, dynamic=False)
        result_compiled = compiled_module(*input_args)

        if isinstance(result_direct, (list, tuple)):
            assert len(result_direct) == len(
                result_compiled
            ), f"Output length mismatch: {len(result_direct)} vs {len(result_compiled)}"

            for i, (direct, compiled) in enumerate(zip(result_direct, result_compiled)):
                assert is_similar(
                    direct.cpu(), compiled.cpu(), rtol=1e-2, atol=1e-2
                ), f"Output {i} mismatch: max diff = {torch.max(torch.abs(direct - compiled)).item()}"
        else:
            assert is_similar(
                result_direct.cpu(), result_compiled.cpu(), rtol=1e-2, atol=1e-2
            ), f"Result mismatch: max diff = {torch.max(torch.abs(result_direct - result_compiled)).item()}"


class TestFoldNdToNzFormat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.XpuGraph(
            xpu_graph.XpuGraphConfig(
                is_training=False,
                debug=True,
                target=xpu_graph.Target.ascend,
                dump_graph=True,
                freeze=True,
                opt_level=OptLevel.level1,
            )
        )

    def test_npu_quant_matmul_nd_to_nz(self, caplog):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "npu_quant_matmul")

        assert "Inserted NZ format cast for weight" in caplog.text
        assert "Converted tensor to NZ format" in caplog.text
        assert "FoldNdToNzFormat: Inserted format casts for" in caplog.text

    def test_npu_weight_quant_batchmatmul_nd_to_nz(self, caplog):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "npu_weight_quant_batchmatmul")

        assert "Inserted format cast for weight in npu_weight_quant_batchmatmul" in caplog.text

    def test_npu_mla_prolog_nd_to_nz(self, caplog):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "npu_mla_prolog")

        assert "Inserted format cast for weight_dq in npu_mla_prolog" in caplog.text
        assert "Inserted format cast for weight_uq_qr in npu_mla_prolog" in caplog.text
        assert "Inserted format cast for weight_dkv_kr in npu_mla_prolog" in caplog.text

    def test_npu_convert_weight_to_int4pack_nd_to_nz(self, caplog):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "npu_convert_weight_to_int4pack")

        assert "Inserted format cast for weight in npu_convert_weight_to_int4pack" in caplog.text

    def test_npu_grouped_matmul_finalize_routing_nd_to_nz(self, caplog):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "npu_grouped_matmul_finalize_routing")

        assert "Inserted format cast for w weight in torch_npu.npu_grouped_matmul_finalize_routing" in caplog.text

    def test_format_conversion_avoidance(self, caplog):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "format_conversion_avoidance")

        assert "Inserted NZ format cast for weight" in caplog.text
        log_text = caplog.text
        format_cast_count = log_text.count("Inserted format cast for x2 weight in npu_quant_matmul")
        assert format_cast_count >= 1, "Format conversion should be triggered"

    def test_torch_mm_nd_to_nz(self, caplog):
        """Test that torch.ops.aten.mm.default supports NZ format conversion for constant weights"""
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            run_nd_to_nz_test(self.xpu_graph_backend, "torch_mm")

        assert "Inserted format cast for second argument in torch.ops.aten.mm.default" in caplog.text
        assert "Converted tensor to NZ format" in caplog.text
        assert "FoldNdToNzFormat: Inserted format casts for" in caplog.text


if __name__ == "__main__":
    test_instance = TestFoldNdToNzFormat()
    test_instance.setup_class()

    run_nd_to_nz_test(test_instance.xpu_graph_backend, "npu_quant_matmul")
    run_nd_to_nz_test(test_instance.xpu_graph_backend, "torch_mm")

    # (@zhaowenshuo 6.26) As of now, torch NPU only supports quant matmul operators with NZ format weights, so the following operators are temporarily on hold.
    # run_nd_to_nz_test(test_instance.xpu_graph_backend, "npu_convert_weight_to_int4pack")
    # run_nd_to_nz_test(test_instance.xpu_graph_backend, "npu_grouped_matmul_finalize_routing")
    # run_nd_to_nz_test(test_instance.xpu_graph_backend, "format_conversion_avoidance")
    # run_nd_to_nz_test(test_instance.xpu_graph_backend, "npu_mla_prolog")
    # run_nd_to_nz_test(test_instance.xpu_graph_backend, "npu_weight_quant_batchmatmul")
