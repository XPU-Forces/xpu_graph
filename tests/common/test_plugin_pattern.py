from functools import cache
from typing import Callable

import pytest
import torch

import xpu_graph
from xpu_graph import (
    XpuGraph,
    enable_plugin_patterns,
    register_plugin_pattern,
    register_this_as_pattern_constraint,
    register_this_as_plugin_pattern,
)
from xpu_graph.config import Target, XpuGraphConfig
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs


def x():
    return torch.empty(
        (10, 1),
        device=torch.device("cpu"),
        requires_grad=False,
        dtype=torch.float32,
    )


@cache
def pattern_fmt(x, postfix=None):
    return f"{x.__module__}-{x.__name__}{f'-{postfix}' if postfix else ''}"


def pattern_reg_success_log(x, postfix=None):
    return f"Pattern {pattern_fmt(x, postfix)}"


def pattern_dereg_success_log(x, postfix=None):
    return f"Deregister pattern {pattern_fmt(x, postfix)}"


def pattern_changed_success_log(x, postfix=None):
    return f"Pattern.{pattern_fmt(x, postfix)} changed"


def constraint_reg_success_log(x, y, postfix=None):
    return f"Register constraint {x.__name__} for {pattern_fmt(y, postfix)} successfully."


class TestPluginPattern:
    def test_basic(self, caplog):
        with need_xpu_graph_logs():
            with enable_plugin_patterns():

                def replace_add_and_div(x, y):
                    return (torch.matmul(x, y),)

                # @register_this_as_plugin_pattern((x(), x()), replace_add_and_div, Target.none)
                def add_and_div(x, y):
                    return (torch.div(x + y, y),)

                # @register_this_as_pattern_constraint
                def constraints0(x, y):
                    if x.dtype != torch.float32 or y.dtype != torch.float32:
                        return False
                    return True

                register_plugin_pattern(
                    add_and_div,
                    (x(), x()),
                    replace_add_and_div,
                    Target.none,
                    (constraints0,),
                )
                # NOTE(liuyuan): make sure the registration is completed.
                assert pattern_reg_success_log(add_and_div) in caplog.text

                @register_this_as_pattern_constraint
                def constraint1(x, y):
                    if x.shape[0] != 1024:
                        return False
                    return True

                assert constraint_reg_success_log(constraint1, add_and_div) in caplog.text

                def another_one(x, y):
                    return (torch.add(x, y).div(y).to(x.dtype),)

                # WARNING(liuyuan): MUST construct compiler just in time.
                config = XpuGraphConfig(is_training=False, debug=True)
                xpu_graph = XpuGraph(config)
                compiled = torch.compile(another_one, backend=xpu_graph, dynamic=False)

                # NOTE(liuyuan): DO NOT match with cosntraint0
                input_tensor = torch.randint(0, 10, (1024, 1024))
                input_tensor_2 = torch.randint(0, 10, (1024, 1024))
                assert (
                    is_similar(
                        compiled(input_tensor, input_tensor_2)[0],
                        replace_add_and_div(input_tensor, input_tensor_2)[0],
                    )
                    == False
                )
                caplog.clear()

                # NOTE(liuyuan): DO NOT match with cosntraint1
                input_tensor = torch.randn(128, 128)
                input_tensor_2 = torch.randn(128, 128)
                assert (
                    is_similar(
                        compiled(input_tensor, input_tensor_2)[0],
                        replace_add_and_div(input_tensor, input_tensor_2)[0],
                    )
                    == False
                )
                caplog.clear()

                input_tensor = torch.randn(1024, 1024)
                input_tensor_2 = torch.randn(1024, 1024)
                assert (
                    is_similar(
                        compiled(input_tensor, input_tensor_2)[0],
                        replace_add_and_div(input_tensor, input_tensor_2)[0],
                    )
                    == True
                )
                assert pattern_changed_success_log(add_and_div) in caplog.text
            assert pattern_dereg_success_log(add_and_div) in caplog.text

    @pytest.mark.xfail
    def test_register_plugin_pattern(self):
        def replace(x):
            return x

        @register_this_as_plugin_pattern((x(),), replace, Target.none)
        def reduce_f(x):
            orig_dtype = x.dtype
            if orig_dtype == torch.float16:
                x = x.to(torch.float32)
            y = x.sum(dim=-1)
            if orig_dtype == torch.float16:
                y = y.to(torch.float16)
            return y

    def test_shape_op(self, caplog):
        with need_xpu_graph_logs():
            with enable_plugin_patterns():

                def replace_1(x):
                    return x * 2

                @register_this_as_plugin_pattern(
                    (torch.empty(10, 128),), replace_1, Target.none, postfix="VariantPatternLt2048"
                )
                @register_this_as_plugin_pattern(
                    (torch.empty(10, 4096),), replace_1, Target.none, postfix="VariantPatternGe2048"
                )
                def pattern(x):
                    return (x[:, :2048].contiguous(),)

                assert pattern_reg_success_log(pattern) in caplog.text
                assert caplog.text.count(pattern_reg_success_log(pattern)) == 2

                # WARNING(liuyuan): MUST construct compiler just in time.
                config = XpuGraphConfig(is_training=False, debug=True)
                xpu_graph = XpuGraph(config)

                compiled = torch.compile(pattern, backend=xpu_graph, dynamic=False)

                input_tensor = torch.randn(1024, 1024)
                assert is_similar(compiled(input_tensor)[0], replace_1(input_tensor)) == True
                assert pattern_changed_success_log(pattern, "VariantPatternLt2048") in caplog.text
                caplog.clear()

                input_tensor = torch.randn(10, 3072)
                assert is_similar(compiled(input_tensor)[0], replace_1(input_tensor)) == True
                assert pattern_changed_success_log(pattern, "VariantPatternGe2048") in caplog.text

            assert pattern_dereg_success_log(pattern) in caplog.text

    def test_variant_pattern(self, caplog):
        with need_xpu_graph_logs():
            with enable_plugin_patterns():

                def replace(x, y):
                    return x * y

                @register_this_as_plugin_pattern((torch.empty(10, 10), torch.empty(10, 10)), replace, Target.none)
                def pattern(x, y):
                    return (x + y,)

                config = XpuGraphConfig(is_training=False, debug=True)
                xpu_graph = XpuGraph(config)

                compiled = torch.compile(pattern, backend=xpu_graph, dynamic=False)
                input_tensor = torch.randn(1024, 1024)
                assert (
                    is_similar(
                        compiled(input_tensor, input_tensor)[0],
                        replace(input_tensor, input_tensor),
                    )
                    == False
                )
                assert pattern_changed_success_log(pattern) not in caplog.text

                # NOTE(liuyuan): This is the variant pattern, equals to pattern2(x, x)
                @register_this_as_plugin_pattern(
                    (torch.empty(10, 10),),
                    replace,
                    Target.none,
                    argument_elimination=lambda x: ((x, x), {}),
                    postfix="VariantPattern2_with_argument_elimination",
                )
                @register_this_as_plugin_pattern((torch.empty(10, 10), torch.empty(10, 10)), replace, Target.none)
                def pattern2(x, y):
                    return (x - y,)

                config = XpuGraphConfig(is_training=False, debug=True)
                xpu_graph = XpuGraph(config)

                compiled = torch.compile(pattern2, backend=xpu_graph, dynamic=False)
                assert (
                    is_similar(
                        compiled(input_tensor, input_tensor)[0],
                        replace(input_tensor, input_tensor),
                    )
                    == True
                )
                assert pattern_changed_success_log(pattern2, "VariantPattern2_with_argument_elimination") in caplog.text
                assert pattern_changed_success_log(pattern2) not in caplog.text

            for p in (pattern, pattern2):
                assert pattern_dereg_success_log(p) in caplog.text

    def test_shape_op_with_fake_literal(self, caplog):
        with enable_plugin_patterns():

            def replace(x):
                return x * 2

            @register_this_as_plugin_pattern((torch.empty(10, 10),), replace, Target.none)
            def pattern(x):
                x = x.view(-1, 1)  # NOTE(liuyuan): Fake literal, best to be 1.
                return (x,)

            def true_func(x):
                x = x.view(-1, x.shape[-1])
                return (x,)

            def true_func_2(x):
                x = x.view(-1, x.shape[-1], x.shape[0])
                return (x,)

            config = XpuGraphConfig(is_training=False, debug=True)
            xpu_graph = XpuGraph(config)
            # NOTE(liuyuan): expectedly, the pattern will be matched.
            compiled = torch.compile(true_func, backend=xpu_graph, dynamic=False)
            # NOTE(liuyuan): expectedly, the pattern will be MISmatched.
            compiled_2 = torch.compile(true_func_2, backend=xpu_graph, dynamic=False)

            input_tensor = torch.rand(1024, 1024 * 1024)
            assert is_similar(compiled(input_tensor)[0], replace(input_tensor))
            assert is_similar(compiled_2(input_tensor)[0], replace(input_tensor)) == False

    def test_pattern_with_control_op(self, caplog):
        with enable_plugin_patterns():

            def x():
                return torch.empty(
                    (1024, 1024),
                    device=torch.device("cpu"),
                    requires_grad=False,
                    dtype=torch.float16,
                )

            def w():
                return torch.empty(
                    (1024,),
                    device=torch.device("cpu"),
                    requires_grad=False,
                    dtype=torch.float32,
                )

            def b():
                return torch.empty(
                    (1024,),
                    device=torch.device("cpu"),
                    requires_grad=False,
                    dtype=torch.float32,
                )

            def replace_layer_norm_2(x):
                return x * 2

            def replace_layer_norm(x, weight, bias):
                return x + weight + bias

            @register_this_as_plugin_pattern(
                (x(), None, None), replace_layer_norm_2, Target.none, postfix="w/o_weight_and_bias"
            )
            @register_this_as_plugin_pattern(
                (x(), w(), w()), replace_layer_norm, Target.none, postfix="w_weight_and_bias"
            )
            def layer_norm(x, weight, bias):
                eps = 1e-5
                x = x.float()
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                x_norm = (x - mean) / torch.sqrt(var + eps)
                if weight is not None and bias is not None:
                    x_norm = weight * x_norm + bias
                x_norm_fp16 = x_norm.half()
                return (x_norm_fp16,)

            @register_this_as_pattern_constraint
            def constraints0(x, weight, bias):
                eps = 1e-5
                if x.dtype != torch.float16:
                    return False
                return True

            def another_one(x, weight, bias):
                eps = 1e-5
                x = x.float()
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                x_norm = (x - mean) / torch.sqrt(var + eps)
                x_norm = weight * x_norm + bias
                x_norm_fp16 = x_norm.half()
                return (x_norm_fp16,)

            def the_other(x):
                eps = 1e-5
                x = x.float()
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                x_norm = (x - mean) / torch.sqrt(var + eps)
                x_norm_fp16 = x_norm.half()
                return (x_norm_fp16,)

            def all_in_one(x, weight, bias):
                x = another_one(x, weight, bias)[0]
                # WARNING(liuyuan); to satisfy the constraint.
                x = x.to(torch.float16)
                return the_other(x)

            config = XpuGraphConfig(is_training=False, debug=True, target=Target.none)
            self.xpu_graph = XpuGraph(config)

            input_tensor = torch.randn((1024, 1024)).half()
            w = torch.randn((1024,)).float()
            b = torch.randn((1024,)).float()

            compiled = torch.compile(all_in_one, backend=self.xpu_graph, dynamic=False)

            assert is_similar(
                compiled(input_tensor, w, b)[0],
                replace_layer_norm_2(replace_layer_norm(input_tensor, w, b).to(torch.float16)),
            )
