import pytest
import torch
import torch_mlu

import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
aten = torch.ops.aten


def fn0(x):
    parts = torch.split(x, 3, dim=1)
    return parts[0], parts[1], parts[2], parts[3]


def split_test(xpu_graph_backend, func):
    for batch in (1,):
        a = torch.randn(batch, 3 * 4, device=device)
        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res_ref = func(a)
        res = compiled(a)
        for i in range(4):
            assert torch.equal(res[i].cpu().float(), res_ref[i].cpu().float())


class TestSplit:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, freeze=True, opt_level=OptLevel.level1)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_split_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            split_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedSplit changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
        opt_level=OptLevel.level1,
        debug=True,
        vendor_compiler_config=None,
    )
    split_test(xpu_graph_backend, fn0)
