import torch
import torch_mlu

import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

device = "mlu:0"
aten = torch.ops.aten
import pytest


def fn0(input_list):
    cat_4 = torch.cat(
        input_list,
        dim=-1,
    )

    stack_4 = torch.stack(input_list)

    return (cat_4, stack_4)


def fn1(input_list):
    cat_4 = torch.cat(
        input_list,
        dim=0,
    )

    stack_4 = torch.stack(input_list)

    return (cat_4, stack_4)


def stack_test(xpu_graph_backend, func):
    # for batch in (10, 512, 31):
    for batch in (31,):
        input_list = [torch.rand(batch, 86).to(device=device) for i in range(5)]
        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res = compiled(input_list)
        res1 = func(input_list)
        for i in range(len(res)):
            if isinstance(res[i], torch.Tensor):
                assert is_similar(res[i].cpu().float(), res1[i].cpu().float())


class TestStackToCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            opt_level=OptLevel.level2,
            debug=False,
            vendor_compiler_config={"mode": "default", "cpp_wrapper": True},
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            stack_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.CatToStack changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
        opt_level=OptLevel.level2,
        debug=True,
        vendor_compiler_config={"mode": "default", "cpp_wrapper": True},
    )
    stack_test(xpu_graph_backend, fn0)
    # stack_test(xpu_graph_backend, fn1)
