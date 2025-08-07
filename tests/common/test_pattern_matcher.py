import operator

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx

from xpu_graph.passes.patterns.utils.pattern_matcher import (
    FxCapture,
    TreeCaptureWrapper,
    add_like,
    dtype_cast_like,
    matmul_like,
)


class TestBuildCaptureWithMetacheck:
    def setup_method(self):
        def check_dtype(node):
            # print(node.format_node() if isinstance(node, fx.Node) else node)
            if not isinstance(node, fx.Node) or node.meta["tensor_meta"].dtype == torch.float32:
                return False
            return True

        def pattern():
            m_input = FxCapture.symbol("input").with_check(
                lambda captured: "input" in captured and check_dtype(captured["input"])
            )
            m_weight = FxCapture.any_if(check_dtype)
            return matmul_like(m_input | dtype_cast_like(m_input), m_weight | dtype_cast_like(m_weight))

        self.capture = pattern()

    def test_capture_structure(self):
        capture_str = str(self.capture)
        assert capture_str.count("call_function[target=aten.matmul.default]") == 1
        assert capture_str.count("call_function[target=aten.mm.default]") == 1
        assert capture_str.count("call_function[target=aten._to_copy.default]") == 2
        assert capture_str.count("?input") == 1
        assert capture_str.count(":= _") == 1
        assert capture_str.count("||") == 5
        assert capture_str.count("meta_check") == 1
        assert capture_str.count("|=") == 1

    def test_capture_graph(self):
        def foo(x, y):
            return torch.matmul(torch.matmul(x, y).to(torch.int32), torch.matmul(x, y).to(torch.int32))

        graph = make_fx(foo)(torch.empty(1, 1024), torch.empty(1024, 1)).graph

        cnt = 0

        for node in reversed(graph.nodes):
            ctx = self.capture.try_capture(node)
            if ctx.status:
                cnt += 1
        assert cnt == 1


class TestBuildCaptureWithLiteral:
    def setup_method(self):
        self.pattern = matmul_like(
            add_like(
                FxCapture.any(),
                FxCapture.symbol("bias1"),
            ),
            add_like(
                FxCapture.any(),
                FxCapture.symbol("bias2"),
            ),
        )

    def test_capture_structure(self):
        capture_str = str(self.pattern)
        assert capture_str.count("call_function[target=aten.matmul.default]") == 1
        assert capture_str.count("call_function[target=aten.mm.default]") == 1
        assert capture_str.count("call_function[target=aten.add.Tensor]") == 2
        assert capture_str.count(":= _") == 2
        assert capture_str.count("||") == 3
        assert capture_str.count("?bias1") == 1
        assert capture_str.count("?bias2") == 1

    def test_capture_graph(self):
        def test(x, y):
            return torch.matmul(x + 2, y + 2)

        graph = make_fx(test)(torch.empty(1, 1024), torch.empty(1024, 1)).graph

        cnt = 0

        for node in reversed(graph.nodes):
            ctx = self.pattern.try_capture(node)
            if ctx.status:
                cnt += 1

        assert cnt == 1


class TestTreeCaptureWrapper:
    def setup_method(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode()

        with fake_mode:
            a = torch.empty(2, 2)
            gamma = torch.empty(2, requires_grad=True)
            beta = torch.empty(2, requires_grad=True)

        m_a = TreeCaptureWrapper.from_tensor(a, "a")
        m_gamma = TreeCaptureWrapper.from_tensor(gamma, "gamma")
        m_beta = TreeCaptureWrapper.from_tensor(beta, "beta")
        m_eps = TreeCaptureWrapper.from_literal(1e-6, "eps")

        def layer_norm(x, gamma, beta, eps):
            mean = torch.mean(x, dim=-1, keepdim=True)
            var = torch.var(x, dim=-1, keepdim=True, unbiased=True)
            y = (x - mean) / torch.sqrt(var + eps)
            norm = y * gamma + beta
            return norm

        self.pattern = layer_norm(m_a, m_gamma, m_beta, m_eps)

    def test_capture_structure(self):
        m_a = FxCapture.symbol("a")
        m_gamma = FxCapture.symbol("gamma")
        m_beta = FxCapture.symbol("beta")
        m_eps = FxCapture.symbol("eps")
        aten = torch.ops.aten
        m_mean = FxCapture.call_function(aten.mean.dim, m_a, [-1], True)
        m_var = FxCapture.call_function(aten.var.correction, m_a, [-1], correction=True, keepdim=True)
        m_sub = FxCapture.call_function(aten.sub.Tensor, m_a, m_mean)
        m_add = FxCapture.call_function(aten.add.Tensor, m_var, m_eps)
        m_sqrt = FxCapture.call_function(aten.sqrt.default, m_add)
        m_div = FxCapture.call_function(aten.div.Tensor, m_sub, m_sqrt)
        m_mul = FxCapture.call_function(aten.mul.Tensor, m_div, m_gamma)
        m_ln = FxCapture.call_function(aten.add.Tensor, m_mul, m_beta)
        assert str(m_ln) == str(self.pattern.matcher)

    def test_capture_literal(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode()
        with fake_mode:
            a = torch.empty(4, 16)
            gamma = torch.empty(16, requires_grad=True)
            beta = torch.empty(16, requires_grad=True)
            eps = 1e-3

        def layer_norm_residual(x, gamma, beta, eps):
            mean = torch.mean(x, dim=-1, keepdim=True)
            var = torch.var(x, dim=-1, keepdim=True, unbiased=True)
            y = (x - mean) / torch.sqrt(var + eps)
            norm = y * gamma + beta
            return x + norm

        gm = make_fx(layer_norm_residual)(a, gamma, beta, eps)

        for node in gm.graph.nodes:
            ctx = self.pattern.try_capture(node)
            if ctx.status:
                assert ctx["eps"] == eps


class TestTreeCaptureWrapperWithDispatch:
    def setup_method(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode()

        with fake_mode:
            a = torch.empty(2, 2)
            gamma = torch.empty(2, requires_grad=True)
            beta = torch.empty(2, requires_grad=True)

        m_a = TreeCaptureWrapper.from_tensor(a, "a")
        m_gamma = TreeCaptureWrapper.from_tensor(gamma, "gamma")
        m_beta = TreeCaptureWrapper.from_tensor(beta, "beta")
        m_eps = TreeCaptureWrapper.from_literal(1e-6, "eps")
        m_shape = TreeCaptureWrapper.from_literal(2, "shape")

        def layer_norm2(x, gamma, beta, eps):
            norm = torch.nn.functional.layer_norm(x, [m_shape], gamma, beta, eps)
            return norm

        self.pattern = layer_norm2(m_a, m_gamma, m_beta, m_eps)

    def test_capture_structure(self):
        m_a = FxCapture.symbol("a")
        m_gamma = FxCapture.symbol("gamma")
        m_beta = FxCapture.symbol("beta")
        m_eps = FxCapture.symbol("eps")
        aten = torch.ops.aten
        m_shape = FxCapture.symbol("shape")
        m_ln = FxCapture.call_function(aten.native_layer_norm.default, m_a, [m_shape], m_gamma, m_beta, m_eps)
        m_out = FxCapture.call_function(operator.getitem, m_ln, 0)
        assert str(self.pattern.matcher) == str(m_out)

    def test_capture_literal(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        fake_mode = FakeTensorMode()
        with fake_mode:
            a = torch.empty(4, 16)
            gamma = torch.empty(16, requires_grad=True)
            beta = torch.empty(16, requires_grad=True)
            eps = 1e-3

        def layer_norm_residual(x, gamma, beta, eps):
            norm = torch.nn.functional.layer_norm(x, gamma.shape, gamma, beta, eps)
            return x + norm

        gm = make_fx(layer_norm_residual)(a, gamma, beta, eps)

        for node in gm.graph.nodes:
            ctx = self.pattern.try_capture(node)
            if ctx.status:
                assert ctx["shape"] == 16
                assert ctx["eps"] == eps


class TestTreeCaptureProduct:
    def setup_method(self):
        def x():
            return torch.empty(
                (10, 1),
                device=torch.device("cpu"),
                requires_grad=False,
                dtype=torch.float32,
            )

        def test_func(x, y, z):
            return x + y, y + z, z + x

        self.gm = make_fx(test_func)(x(), x(), x())

        aten = torch.ops.aten
        add_pat1 = FxCapture.call_function(aten.add.Tensor, (FxCapture.symbol("x"), FxCapture.symbol("y")))
        add_pat2 = FxCapture.call_function(aten.add.Tensor, (FxCapture.symbol("y"), FxCapture.symbol("z")))
        self.pattern = FxCapture.product(add_pat1, add_pat2)

    def test_capture_product(self):
        cnt = 0
        for _ in self.pattern.iterate_all_captured(self.gm.graph.nodes):
            cnt += 1
        assert cnt == 3
