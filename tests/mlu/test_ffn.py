import pytest

import torch
import torch_mlu
import xpu_graph

from xpu_graph.test_utils import is_similar

import torch_mlu_ops as ops


aten = torch.ops.aten
device = "mlu:0"
data_type = torch.float32

act_mode_dict = {
    "relu": torch.nn.functional.relu,
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
}


class FeedForward(torch.nn.Module):
    def __init__(
        self, input_size: int, inner_size: int, act_mode: str, bias=False, gated=False
    ):
        super(FeedForward, self).__init__()
        self.up_linear = torch.nn.Linear(input_size, inner_size, bias)
        self.gated = gated
        if self.gated:
            self.gated_linear = torch.nn.Linear(input_size, inner_size, bias)
        self.down_linear = torch.nn.Linear(inner_size, input_size, bias)
        self.act = act_mode_dict[act_mode]

    def forward(self, x):
        act_out = self.act(self.up_linear(x).float()).to(x.dtype)
        return (
            self.down_linear(act_out * self.gated_linear(x))
            if self.gated
            else self.down_linear(act_out)
        )


def fn0(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = torch.matmul(ffn_input, ffn_weight1.transpose(1, 0))
    if bias1 is not None:
        output = output + bias1
    torch_act = act_mode_dict[act]
    output = torch_act(output)
    output = torch.matmul(output, ffn_weight2.transpose(1, 0))
    if bias2 is not None:
        output = output + bias2
    return output


def fn1(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = fn0(ffn_input, ffn_weight1, ffn_weight2, bias1, bias2, act)
    return output


def fn2(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = fn0(ffn_input, ffn_weight1, ffn_weight2, bias1, bias2, act)
    return output


def fn3(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = torch.matmul(ffn_input, ffn_weight1.transpose(1, 0))
    if bias1 is not None:
        output = output + bias1
    torch_act = act_mode_dict[act]
    output = torch_act(output)
    output = torch.matmul(output, ffn_weight2.transpose(1, 0))
    if bias2 is not None:
        output = output + bias2
    return output.view(-1)


def ffn_test(xpu_graph_backend, func):
    with torch.no_grad():
        batch = 5
        seq_len = 16
        input_size = 512
        hidden_size = 1024
        dtype = torch.float32
        if func in [fn0, fn1]:
            act_mode = "silu"
        else:
            act_mode = "gelu"
        if func in [fn0]:
            bias = False
        else:
            bias = True

        use_gate = False
        ffn_input = torch.randn(
            (batch, seq_len, input_size), dtype=dtype, device=device
        )
        args = [ffn_input]
        pytorch_ffn = FeedForward(input_size, hidden_size, act_mode, bias=bias).mlu()
        pytorch_ffn = pytorch_ffn.to(dtype)
        args += [
            pytorch_ffn.up_linear.weight,
            pytorch_ffn.down_linear.weight,
            pytorch_ffn.up_linear.bias,
            pytorch_ffn.down_linear.bias,
        ]
        args += [act_mode]

        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res1 = func(*args)
        res = compiled(*args)
        assert is_similar(res1.float(), res.float())
        tmo_output1 = ops.ffn(
            ffn_input,
            pytorch_ffn.up_linear.weight,
            pytorch_ffn.up_linear.bias,
            pytorch_ffn.down_linear.weight,
            pytorch_ffn.down_linear.bias,
            pytorch_ffn.gated_linear.weight if use_gate else None,
            pytorch_ffn.gated_linear.bias if use_gate else None,
            act_mode,
        )
        assert is_similar(res1.float().reshape(-1), tmo_output1.float().reshape(-1))


class TestFFN:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler()

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3],
    )
    def test_slice_patterns(self, pattern_func):
        ffn_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    pytest.main()
