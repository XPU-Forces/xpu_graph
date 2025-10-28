import pytest
import torch
import torch._dynamo.config as dynamo_config

import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn_view(inp, mask, query):
    """
    This is a mimic process of unpadding + target_attention + padding
    Assure capture_dynamic_output_shape_ops=True for putting dynamic-shape in FxGraph
    """
    bsz, seq, dim = inp.shape
    qnum = query.shape[0]

    # 0. concat the query token and input token, and their padding masks
    cat_seq = torch.cat([query.unsqueeze(0).expand(bsz, -1, -1), inp], dim=1)
    qmask = torch.ones([bsz, qnum], device=inp.device, dtype=torch.int32)
    cat_mask = torch.cat([qmask, mask], dim=1)

    # 1. the unpadding part, reshape the padding mask 1D as the select mask, concat all unpadded sequences allong the seq dim
    zmask = torch.reshape(cat_mask, [-1])
    zindices = torch.nonzero(zmask)
    out = torch.reshape(cat_seq, [-1, cat_seq.size(2)])[zindices].squeeze(1)

    # 2. unpadding the source mask as well (0 for query token, 1 for input token), to extract target tokens later
    mask0 = torch.zeros([bsz, qnum], device=inp.device, dtype=torch.int32)
    mask1 = torch.ones([bsz, seq], device=inp.device, dtype=torch.int32)
    source_mask = torch.cat([mask0, mask1], dim=1)
    unpad_out_mask = torch.reshape(source_mask, [-1])[zindices].squeeze(1)

    # 3. (skip the attention part). Extract the query token output.
    query_pos = torch.nonzero(torch.eq(unpad_out_mask, 0))
    #   By human knowledge we known the query mask is the same as the num of query tokens, but
    #   dynamo cannot infer this out directly. Thus, this reshape performs as an assertion
    query_out = out[query_pos].squeeze(1).reshape(bsz, qnum, dim)

    return query_out.view(-1, qnum * dim)


@dynamo_config.patch(capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True)
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
    def test_symshape_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs():
            fn_test(self.xpu_graph_backend, pattern_func)

        assert "torch.ops.aten.nonzero" in caplog.text
        assert caplog.text.count("Pattern.FoldView1 changed graph") == 2


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False, opt_level=OptLevel.level2, debug=True)
    fn_test(xpu_graph_backend, fn_view)
