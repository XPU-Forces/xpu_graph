import math
import pytest

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import assertTensorsEqual


def fn0(query, key, value, inv_scale=1.0):
    return (
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale)
        .softmax(dim=-1)
        .matmul(value)
    )


def fn1(query, key, value):
    return torch.matmul(query, key.transpose(-2, -1)).softmax(dim=-1)


def fn2(query, key, value):
    query = query.view(-1, query.shape[2], query.shape[3])
    key = key.transpose(-2, -1)
    key = key.view(query.shape[0], query.shape[2], -1)
    return torch.bmm(query, key)


def bmm_test(xpu_graph_backend, func):
    head_size = 64
    seq_q, seq_k = 38, 38
    head_num_q, head_num_k = 32, 32
    dtype = torch.half
    batch = 1
    q = torch.randn(batch, head_num_q, seq_q, head_size, dtype=dtype, device="mlu")
    k = torch.randn(batch, head_num_k, seq_k, head_size, dtype=dtype, device="mlu")
    v = torch.randn(batch, head_num_k, seq_k, head_size, dtype=dtype, device="mlu")

    res1 = func(q, k, v)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res = compiled(q, k, v)

    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


class TestBMM:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2],
    )
    def test_sfdp_patterns(self, caplog, pattern_func):
        bmm_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedBMM changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(freeze=True, opt_level=OptLevel.level2)
    bmm_test(xpu_graph_backend, fn0)
    bmm_test(xpu_graph_backend, fn1)
    bmm_test(xpu_graph_backend, fn2)
