from functools import cache
from typing import Callable

import torch

from xpu_graph import (
    XpuGraph,
    register_plugin_pattern,
    register_this_as_pattern_constraint,
    register_this_as_plugin_pattern,
)
from xpu_graph.config import Target, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def linear_replace(x, w):
    return torch.ops.aten.mm.default(x, w)


@register_this_as_plugin_pattern(
    (torch.empty(10, 10), torch.empty(10, 10)), linear_replace, Target.mlu, ignore_literal=True
)
def linear_pattern1(x, w):
    return (torch.nn.functional.linear(x, w.T),)


def linear_bias_replace(x, w, b):
    return torch.ops.aten.addmm.default(b, x, w)


@register_this_as_plugin_pattern(
    (
        torch.empty(10, 10),
        torch.empty(10, 10),
        torch.empty(
            10,
        ),
    ),
    linear_bias_replace,
    Target.mlu,
    ignore_literal=True,
)
def linear_pattern2(x, w, b):
    return (torch.nn.functional.linear(x, w.T, b),)
