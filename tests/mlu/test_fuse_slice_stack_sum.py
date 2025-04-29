import pytest

import torch
import torch_mlu
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)
from xpu_graph.test_utils import is_similar


device = "mlu:0"
aten = torch.ops.aten


def fn0(arg41_1):
    slice_190 = arg41_1[:, 35263:35295]
    slice_192 = arg41_1[:, 35296:35328]
    slice_194 = arg41_1[:, 35329:35361]
    slice_196 = arg41_1[:, 35378:35410]
    slice_198 = arg41_1[:, 35411:35443]
    slice_200 = arg41_1[:, 35444:35476]
    slice_202 = arg41_1[:, 35971:36003]
    slice_204 = arg41_1[:, 37206:37238]
    slice_206 = arg41_1[:, 37256:37288]
    slice_208 = arg41_1[:, 37306:37338]
    slice_210 = arg41_1[:, 37391:37423]
    slice_212 = arg41_1[:, 37432:37464]
    slice_214 = arg41_1[:, 37474:37506]
    slice_216 = arg41_1[:, 1165:1197]
    slice_218 = arg41_1[:, 1340:1372]
    slice_220 = arg41_1[:, 1387:1419]
    slice_222 = arg41_1[:, 1739:1771]
    slice_224 = arg41_1[:, 2629:2661]
    slice_226 = arg41_1[:, 2676:2708]
    slice_228 = arg41_1[:, 2723:2755]
    slice_230 = arg41_1[:, 2897:2929]
    slice_232 = arg41_1[:, 2186:2218]
    slice_234 = arg41_1[:, 2232:2264]
    slice_236 = arg41_1[:, 2278:2310]
    slice_238 = arg41_1[:, 2417:2449]

    stack_4 = torch.stack(
        [
            slice_190,
            slice_192,
            slice_194,
            slice_196,
            slice_198,
            slice_200,
            slice_202,
            slice_204,
            slice_206,
            slice_208,
            slice_210,
            slice_212,
            slice_214,
            slice_216,
            slice_218,
            slice_220,
            slice_222,
            slice_224,
            slice_226,
            slice_228,
            slice_230,
            slice_232,
            slice_234,
            slice_236,
            slice_238,
        ],
        dim=0,
    )

    sum_6 = torch.sum(stack_4, dim=0)
    return (sum_6,)


def fn1(x):
    axis = -1
    c = torch.cat(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        axis,
    )
    c = c + 1
    d = torch.stack(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        0,
    )
    return (c, d)


def fn2(x):
    axis = -1
    b = x[:, 39415:39479].contiguous()
    b += 1
    c = aten.stack(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 34065:34129],
            b,
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
    )
    return (c, 1)


def stack_test(xpu_graph_backend, func):
    for batch in (10, 512, 31):
        a = torch.randn(batch, 43106).to(device=device)
        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res = compiled(a)[0]
        res1 = func(a)[0]
        assert is_similar(res.cpu().float(), res1.cpu().float())

class TestSliceStackSum:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            stack_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedSliceStackSum changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=True, opt_level=OptLevel.level2, debug=True
    )
    stack_test(xpu_graph_backend, fn0)
    #stack_test(xpu_graph_backend, fn1)
    #stack_test(xpu_graph_backend, fn2)
