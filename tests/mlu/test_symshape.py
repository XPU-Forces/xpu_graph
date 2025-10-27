import pytest
import torch

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn_view(inp, mask, query):
    bsz, seq, dim = inp.shape
    qnum = query.shape[0]
    qmask = torch.ones([bsz, qnum], device=inp.device, dtype=torch.int32)
    cat_mask = torch.cat([qmask, mask], dim=1)
    cat_seq = torch.cat([query.unsqueeze(0).expand(bsz, -1, -1), inp], dim=1)

    zmask = torch.reshape(cat_mask, [-1])
    zindices = torch.nonzero(zmask)
    out = torch.reshape(cat_seq, [-1, cat_seq.size(2)])[zindices].squeeze(1)

    mask0 = torch.zeros([bsz, qnum], device=inp.device)
    mask1 = torch.ones([bsz, seq], device=inp.device)
    source_mask = torch.cat([mask0, mask1], dim=1)

    out_mask = torch.reshape(source_mask, [-1])[zindices].squeeze(1)
    query_pos = torch.nonzero(torch.eq(out_mask, 0))
    query_out = out[query_pos].reshape(bsz, qnum, dim)

    return query_out.view(-1, qnum * dim)


def fn_test(xpu_graph_backend, func):
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)
    for bsz in [8, 80, 100]:
        inp = torch.randn((bsz, 1024, 32), device=device, dtype=data_type)
        mask = (torch.arange(bsz).unsqueeze(1) >= torch.arange(1024).unsqueeze(0)).to(device).to(torch.int32)
        query = torch.randn(2, 32, device=device, dtype=data_type)
        res = func(inp, mask, query)
        res1 = compiled(inp, mask, query)
        assert is_similar(res.cpu().float(), res1.cpu().float())


class TestSymShape:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn_view,
        ],
    )
    def test_symshape_patterns(self, caplog, monkeypatch, pattern_func):
        import torch._dynamo.config

        monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", True)
        monkeypatch.setattr(torch._dynamo.config, "capture_dynamic_output_shape_ops", True)
        with need_xpu_graph_logs():
            fn_test(self.xpu_graph_backend, pattern_func)

        assert caplog.text.count("Pattern.FoldView1 changed graph") == 2


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, debug=True)
    import torch._dynamo.config

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    fn_test(xpu_graph_backend, fn_view)
