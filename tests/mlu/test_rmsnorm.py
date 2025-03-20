import pytest
import torch
import torch_mlu
import xpu_graph
from xpu_graph.test_utils import is_similar
from xpu_graph.config import OptLevel


class RMSNorm1(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNorm2(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype) * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


fn0 = RMSNorm1(hidden_size=(10,)).mlu()
fn1 = RMSNorm2(hidden_size=(10,)).mlu()


def fn2(hidden_states):
    residual = hidden_states.clone()
    input_ = hidden_states + residual
    output = fn0(input_)
    return output


def fn3(hidden_states):
    residual = hidden_states.clone()
    input_ = hidden_states + residual
    output = fn0(input_)
    return output, input_


def rmsnorm_test(xpu_graph, func):
    with torch.no_grad():
        a = torch.randn(1, 10).mlu()
        compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
        if func != fn3:
            norm = compiled(a)
            norm1 = func(a)
            assert is_similar(norm1.float(), norm.float())
        else:
            norm, res = compiled(a)
            norm1, res1 = func(a)
            assert is_similar(norm1.float(), norm.float())
            assert is_similar(res1.float(), res.float())


class TestRMSNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
        ],
    )
    def test_rmsnorm_patterns(self, pattern_func):
        rmsnorm_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level2
    )
    rmsnorm_test(xpu_graph_backend, fn0)
    rmsnorm_test(xpu_graph_backend, fn1)
    rmsnorm_test(xpu_graph_backend, fn2)
    rmsnorm_test(xpu_graph_backend, fn3)
