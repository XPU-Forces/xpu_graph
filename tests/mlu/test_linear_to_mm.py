from functools import cache
from typing import Callable

import pytest
import torch

import xpu_graph
from xpu_graph import (
    XpuGraph,
    register_plugin_pattern,
    register_this_as_pattern_constraint,
    register_this_as_plugin_pattern,
)
from xpu_graph.config import Target, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


class TestPluginPattern:
    def test_linear(self, caplog):
        xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            debug=True,
            vendor_compiler_config=None,
        )

        def test_func(x, w):
            return torch.nn.functional.linear(x, w.T)


        compiled = torch.compile(test_func, backend=xpu_graph_backend, dynamic=False)

        input_tensor = torch.rand(1024, 1024).to("mlu")
        weight_tensor = torch.rand(1024, 1024).to("mlu")

    def test_linear_bias(self, caplog):
        xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False,
            debug=True,
            vendor_compiler_config=None,
        )

        def test_func(x, w, b):
            return torch.nn.functional.linear(x, w.T, b)


        compiled = torch.compile(test_func, backend=xpu_graph_backend, dynamic=False)

        input_tensor = torch.rand(1024, 1024).to("mlu")
        weight_tensor = torch.rand(1024, 1024).to("mlu")
        bias_tensor = torch.rand(1024).to("mlu")
        assert is_similar(compiled(input_tensor, weight_tensor, bias_tensor), replace(input_tensor, weight_tensor))
