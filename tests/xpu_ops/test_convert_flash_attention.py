import math

import pytest
import torch
import torch_npu
import xpu_ops

import xpu_graph
from xpu_graph.compiler import XpuGraph
from xpu_graph.config import XpuGraphConfig

xpu_ops.load_xpu_ops_npu()


def test_flash_attention():
    batch = 1
    q_head_num = 8
    kv_head_num = 1
    q_seq_length = 1 * 1024
    kv_seq_length = 1 * 1024
    head_dim = 128

    def _attention(query, key, value):
        mask = 1 - torch.tril(torch.ones(q_seq_length, kv_seq_length, dtype=torch.uint8)).npu()
        scale = 1.0 / math.sqrt(head_dim)
        output = torch_npu.npu_prompt_flash_attention(
            query,
            key,
            value,
            num_heads=8,
            num_key_value_heads=1,
            input_layout="BNSD",
            atten_mask=mask,
            scale_value=scale,
            pre_tokens=65535,
            next_tokens=0,
        )
        return output

    query = torch.randn(batch, q_head_num, q_seq_length, head_dim).to(dtype=torch.bfloat16).npu()
    key = torch.randn(batch, kv_head_num, kv_seq_length, head_dim).to(dtype=torch.bfloat16).npu()
    value = torch.randn(batch, kv_head_num, kv_seq_length, head_dim).to(dtype=torch.bfloat16).npu()

    config = XpuGraphConfig(target=xpu_graph.config.Target.npu, vendor_compiler_config={"compiler": "ge"})
    compiled = torch.compile(_attention, backend=XpuGraph(config), dynamic=False)

    res = compiled(query, key, value)

    from xpu_graph.test_utils import is_similar

    assert is_similar(res.cpu(), _attention(query, key, value).cpu())
