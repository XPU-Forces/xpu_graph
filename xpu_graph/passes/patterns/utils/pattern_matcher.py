import operator
from typing import Callable

import torch
from torch.fx import Node as FxNode
from torch.utils._pytree import tree_iter, tree_map, tree_map_only


class CaptureResult:
    def __init__(self):
        self.status = True
        self.symbols = {}
        self.captured_nodes = {}
        self.panic_msg = ""
        self.root_capture = None

    def panic(self, msg=""):
        self.panic_msg = msg
        self.status = False
        return False

    def restore_state(self, state):
        self.status, self.panic_msg, captured_symbols, captured_nodes_keys = state
        self.symbols = {k: self.symbols[k] for k in captured_symbols}
        self.captured_nodes = {k: self.captured_nodes[k] for k in captured_nodes_keys}

    def backup_state(self):
        return (self.status, self.panic_msg, set(self.symbols.keys()), set(self.captured_nodes.keys()))

    def match_symbol(self, key, val):
        if not self.status:
            return
        if key is None:
            return
        if key in self.symbols:
            if not self.symbols[key] == val:
                self.panic(f"Key {key.symbol} already captured with value {self.symbols[key]}, but we got {val} now")
        else:
            self.symbols[key] = val

    def match_val(self, target, val):
        if not self.status:
            return
        if not target == val:
            self.panic(f"Value {val} not match expected {target}")

    def match_fxnode(self, node_capture, node):
        if not self.status:
            return
        if not isinstance(node, FxNode):
            self.panic(f"Node {node} is not a Fx Node")
            return
        if node_capture.op != node.op:
            self.panic(f"Node op {node.op} not match expected {node_capture.op}")
        if node_capture.target != node.target:
            self.panic(f"Node target {node.target} not match expected {node_capture.target}")
        for sub_capture, sub_node in zip(
            tree_iter((node_capture.args, node_capture.kwargs)), tree_iter((node.args, node.kwargs))
        ):
            if not self.status:
                return
            if isinstance(sub_capture, FxCapture):
                if not self.match_capture(sub_capture, sub_node):
                    return
            else:
                self.match_val(sub_capture, sub_node)

    def match_union(self, candidates, node: object):
        state = self.backup_state()
        fail_msgs = []
        for candidate in candidates:
            if self.match_capture(candidate, node):
                return
            else:
                fail_msgs.append(self.panic_msg)
                self.restore_state(state)
        self.panic("All candidates failed:\n    " + "\n && ".join(msg.replace("\n", "\n    ") for msg in fail_msgs))

    def match_product(self, candidates, nodes: object):
        if not isinstance(nodes, (tuple, list)):
            self.panic(f"Nodes {nodes} is not a tuple or list")
            return
        if len(candidates) != len(nodes):
            self.panic(f"Nodes {nodes} and candidates {candidates} have different length")
            return
        for candidate, node in zip(candidates, nodes):
            self.match_capture(candidate, node)

    def match_capture(self, capture, node_or_val: object) -> bool:
        if self.root_capture is None:
            self.root_capture = capture
        if not self.status:
            return False
        if capture in self.captured_nodes:
            self.match_val(self.captured_nodes[capture], node_or_val)
        elif capture.op == "union":
            self.match_union(capture.args, node_or_val)
        elif capture.op == "product":
            self.match_product(capture.args, node_or_val)
        elif capture.op == "predicate":
            if self.match_capture(capture.args[0], node_or_val):
                if not capture.target(self.symbols):
                    self.panic(
                        f"Meta check for partial captured results failed for already captured:\n"
                        + "\n  ".join(
                            f"{k}: {v.format_node() if isinstance(v, FxNode) else v}" for k, v in self.symbols.items()
                        )
                    )
        elif capture.op == "symbol":
            self.match_symbol(capture.target, node_or_val)
            if self.status and "predicate" in capture.kwargs:
                try:
                    result = capture.kwargs["predicate"](node_or_val)
                except:
                    result = False
                if not result:
                    self.panic(
                        f"Predicate failed for {node_or_val.format_node() if isinstance(node_or_val, FxNode) else node_or_val}"
                    )
        elif capture.op in ["call_function", "call_module", "call_method", "get_attr"]:
            self.match_fxnode(capture, node_or_val)
        else:
            raise NotImplementedError(f"Unsupported capture type: {type(capture)}")

        if self.status:
            self.captured_nodes[capture] = node_or_val
        return self.status

    def __getitem__(self, id):
        if isinstance(id, str):
            return self.symbols[id]
        elif isinstance(id, FxCapture):
            return self.captured_nodes[id]

    def __str__(self):
        if self.root_capture is None:
            return "Unused"
        if self.status:
            s = "Capture success: "
            s += "\n  symbols:"
            for k, v in self.symbols.items():
                s += f"    {k}: {v}"
            s += f"  captured:"

            def captured_result_repr(node, node_id_map, repr_list):
                single_line_repr = node.single_line_repr(node_id_map, repr_list)
                if node in ctx.captured_nodes:
                    captuerd_node = ctx[node]
                    captuerd_node_repr = (
                        captuerd_node.format_node() if isinstance(captuerd_node, FxNode) else captuerd_node
                    )
                    str = f"    {single_line_repr}\n      {captuerd_node_repr}"
                else:
                    str = None
                return str

            repr_list = self.root_capture.to_list(captured_result_repr)
            for line in repr_list:
                if line is not None:
                    s += "\n" + line
            return s
        else:
            return f"Capture failed: {ctx.panic_msg}"


class FxCapture:
    def __init__(self, op, target, args, kwargs):
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def __or__(self, capture):
        return __class__.union(self, capture)

    def __getitem__(self, id):
        return __class__.call_function(operator.getitem, (self, id))

    def with_check(self, predicate):
        return __class__.predicate(self, predicate)

    def try_capture(self, node_or_val):
        ctx = CaptureResult()
        result = ctx.match_capture(self, node_or_val)
        return ctx

    def replace(self, tracked_literal_to_wrapper):
        def inner_replace(t):
            if t in tracked_literal_to_wrapper:
                t = tracked_literal_to_wrapper[t]
            elif isinstance(t, FxCapture):
                t = t.replace(tracked_literal_to_wrapper)
            return t

        args, kwargs = tree_map(inner_replace, (self.args, self.kwargs))
        return __class__(self.op, self.target, args, kwargs)

    def single_line_repr(self, node_id_map, result_list):
        if self.op == "symbol":
            if self.target is not None:
                expr = "?" + self.target
            else:
                expr = "_"
            if "predicate" in self.kwargs:
                expr += f" |= [...]"

        elif self.op == "union":
            expr = " || ".join(f"%{node_id_map[cand]}" for cand in self.args)
        elif self.op == "product":
            expr = "(" + ", ".join(f"%{node_id_map[cand]}" for cand in self.args) + ")"
        elif self.op == "predicate":
            expr = f"meta_check(...)"
        elif self.op in ["call_function", "call_method", "call_module", "get_attr"]:

            class Numbering:
                def __init__(self, id):
                    self.id = id

                def __str__(self):
                    return f"%{self.id}"

                def __repr__(self):
                    return str(self)

            def map_capture_to_id(obj):
                return Numbering(node_id_map[obj])

            expr = f"{self.op}[target={self.target}](args={tree_map_only(FxCapture, map_capture_to_id, self.args)}, kwargs={tree_map_only(FxCapture, map_capture_to_id, self.kwargs)})"
        else:
            raise ValueError(f"Unsupported op: {self.op}")
        return f"%{len(result_list)} := {expr}"

    def to_list(self, map_fn=None, node_id_map=None, result_list=None):
        if node_id_map is None:
            node_id_map = {}
        if result_list is None:
            result_list = []
        if self in node_id_map:
            return result_list
        for sub_node in tree_iter((self.args, self.kwargs)):
            if isinstance(sub_node, __class__):
                result_list = sub_node.to_list(map_fn, node_id_map, result_list)
        result = map_fn(self, node_id_map, result_list)
        node_id_map[self] = len(result_list)
        result_list.append(result)
        return result_list

    def __str__(self):
        repr_list = self.to_list(lambda node, node_id_map, result_list: node.single_line_repr(node_id_map, result_list))

        return "\n".join(repr_list)

    @staticmethod
    def union(*captures):
        candidates = []
        for capture in captures:
            if capture.op == "union":
                candidates.extend(capture.args)
            else:
                candidates.append(capture)

        return __class__("union", None, args=tuple(candidates), kwargs={})

    @staticmethod
    def product(*captures):
        return __class__("product", None, args=tuple(captures), kwargs={})

    @staticmethod
    def predicate(capture, meta_check):
        return __class__("predicate", meta_check, args=(capture,), kwargs={})

    @staticmethod
    def symbol(symbol: str):
        return __class__("symbol", symbol, args=(), kwargs={})

    @staticmethod
    def symbol_if(symbol: str, predicate: Callable[..., bool]):
        return __class__("symbol", symbol, args=(), kwargs={"predicate": predicate})

    @staticmethod
    def any():
        return __class__("symbol", None, args=(), kwargs={})

    @staticmethod
    def any_if(predicate: Callable[..., bool]):
        return __class__("symbol", None, args=(), kwargs={"predicate": predicate})

    @staticmethod
    def call_function(target, *args, **kwargs):
        return __class__("call_function", target, args, kwargs)

    @staticmethod
    def call_method(target, *args, **kwargs):
        return __class__("call_method", target, args, kwargs)

    @staticmethod
    def call_module(target, *args, **kwargs):
        return __class__("call_module", target, args, kwargs)

    @staticmethod
    def get_attr(target):
        return __class__("get_attr", target, (), {})


aten = torch.ops.aten


def matmul_like(m_capture, n_capture):
    if m_capture is None:
        m_capture = FxCapture.any()
    if n_capture is None:
        n_capture = FxCapture.any()
    return FxCapture.call_function(aten.mm.default, m_capture, n_capture) | FxCapture.call_function(
        aten.matmul.default, m_capture, n_capture
    )


def add_like(left_capture, right_capture):
    if left_capture is None:
        left_capture = FxCapture.any()
    if right_capture is None:
        right_capture = FxCapture.any()
    return FxCapture.call_function(aten.add.Scalar, left_capture, right_capture) | FxCapture.call_function(
        aten.add.Tensor, left_capture, right_capture
    )


def dtype_cast_like(capture):
    if capture is None:
        capture = FxCapture.any()
    return FxCapture.call_function(aten._to_copy, capture) | FxCapture.call_function(aten._to_copy.default, capture)


def zero_like():
    return FxCapture.call_function(aten.zeros_like.default, FxCapture.any()) | FxCapture.call_function(
        aten.zeros.default
    )


def one_like():
    return FxCapture.call_function(aten.ones_like.default, FxCapture.any()) | FxCapture.call_function(aten.ones.default)


class TreeCaptureLiteral:
    def __init__(self, fake_elem, matcher):
        self.fake_elem = fake_elem
        self.matcher = matcher

    @classmethod
    def __class_getitem__(cls, fake_cls):
        # Note: TreeCaptureLiteral need to fake itself as a SymFloat/SymInt/SymBool to support builtin ops
        class LiteralWrapper(fake_cls, TreeCaptureLiteral):
            def __init__(self, fake_elem, matcher):
                TreeCaptureLiteral.__init__(self, fake_elem, matcher)

        return LiteralWrapper


class TreeCaptureTensor(torch.Tensor):
    def __new__(cls, fake_elem, matcher):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            fake_elem.size(),
            strides=fake_elem.stride(),
            storage_offset=fake_elem.storage_offset(),
            dtype=fake_elem.dtype,
            layout=fake_elem.layout,
            requires_grad=fake_elem.requires_grad,
            device=fake_elem.device,
        )

    def __init__(self, fake_elem, matcher=None):
        self.fake_elem = fake_elem
        self.matcher = matcher

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                return t.fake_elem
            else:
                return t

        def get_matcher_fn(t):
            if isinstance(t, cls):
                return t.matcher
            elif isinstance(t, torch.Tensor):
                return FxCapture.any()
            else:
                return t

        kwargs = kwargs or {}

        r_matcher = FxCapture.call_function(func, *tree_map(get_matcher_fn, args), **tree_map(get_matcher_fn, kwargs))
        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def gen_getitem_matcher(src, fake_elem, i):
            if isinstance(fake_elem, torch.Tensor):
                return cls(fake_elem, src[i])
            else:
                return fake_elem

        # is generally a safe bet in the current codegen
        if isinstance(r, list):
            return [gen_getitem_matcher(r_matcher, t, i) for i, t in enumerate(r)]
        elif isinstance(r, tuple):
            return tuple(gen_getitem_matcher(r_matcher, t, i) for i, t in enumerate(r))
        elif isinstance(r, torch.Tensor):
            return cls(r, r_matcher)
        else:
            return r

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        tracked_literals = {}

        def unwrap_tracked_literals(t):
            if isinstance(t, TreeCaptureLiteral):
                if t.fake_elem in tracked_literals and tracked_literals[t.fake_elem] != t.matcher:
                    raise ValueError(
                        "Cannot distinguish between wrapped TreeMatchingLiterals, maybe you can try differnet example values"
                    )
                tracked_literals[t.fake_elem] = t.matcher
                return t.fake_elem
            elif not isinstance(t, torch.Tensor):
                tracked_literals[t] = t
                return t
            else:
                return t

        args, kwargs = tree_map(unwrap_tracked_literals, (args, kwargs))
        ret = super().__torch_function__(func, types, args, kwargs)

        def replace_tracked_literal(t):
            if isinstance(t, TreeCaptureTensor):
                t.matcher = t.matcher.replace(tracked_literals)
            return t

        return tree_map(replace_tracked_literal, ret)

    def __str__(self):
        return str(self.matcher) + " | hint=" + str(self.fake_elem)

    def try_capture(self, node):
        return self.matcher.try_capture(node)


class TreeCaptureWrapper:
    @staticmethod
    def from_literal(fake_val, name):
        matcher = FxCapture.symbol(name)
        if isinstance(fake_val, float):
            return TreeCaptureLiteral[torch.SymFloat](fake_val, matcher)
        elif isinstance(fake_val, int):
            return TreeCaptureLiteral[torch.SymInt](fake_val, matcher)
        elif isinstance(fake_val, bool):
            return TreeCaptureLiteral[torch.SymBool](fake_val, matcher)
        else:
            raise ValueError(f"Unsupported literal type: {type(fake_val)}")

    @staticmethod
    def from_tensor(tensor, name):
        return TreeCaptureTensor(tensor, FxCapture.symbol(name))


if __name__ == "__main__":
    import logging

    from torch.fx.experimental.proxy_tensor import make_fx

    from xpu_graph.utils import setup_logger

    setup_logger(logging.DEBUG)

    def check_dtype(node):
        # print(node.format_node() if isinstance(node, FxNode) else node)
        if not isinstance(node, FxNode) or node.meta["tensor_meta"].dtype == torch.float32:
            return False
        return True

    def pattern():
        m_input = FxCapture.symbol("input").with_check(
            lambda captured: "input" in captured and check_dtype(captured["input"])
        )
        m_weight = FxCapture.any_if(check_dtype)
        return matmul_like(m_input | dtype_cast_like(m_input), m_weight | dtype_cast_like(m_weight))

    def foo(x, y):
        return torch.matmul(torch.matmul(x, y).to(torch.int32), torch.matmul(x, y).to(torch.int32))

    graph = make_fx(foo)(torch.empty(1, 1024), torch.empty(1024, 1)).graph

    cnt = 0

    print(graph)
    pat = pattern()
    print(pat)
    for node in reversed(graph.nodes):
        ctx = pat.try_capture(node)
        if ctx.status:
            print(ctx)
            cnt += 1
    print(cnt)
    assert cnt == 1

    def test(x, y):
        return torch.matmul(x + 2, y + 2)

    graph = make_fx(test)(torch.empty(1, 1024), torch.empty(1024, 1)).graph
    print(graph)
    cnt = 0
    pattern = matmul_like(
        add_like(
            FxCapture.any(),
            FxCapture.symbol("bias1"),
        ),
        add_like(
            FxCapture.any(),
            FxCapture.symbol("bias2"),
        ),
    )
    for node in reversed(graph.nodes):
        ctx = pattern.try_capture(node)
        if ctx.status:
            print(ctx)
            cnt += 1
    assert cnt == 1
    print(cnt)
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode()

    with fake_mode:
        a = torch.empty(2, 2, device="mlu")
        gamma = torch.empty(2, device="mlu", requires_grad=True)
        beta = torch.empty(2, device="mlu", requires_grad=True)

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

    m_ln = layer_norm(m_a, m_gamma, m_beta, m_eps)
    print(m_ln)

    m_shape = TreeCaptureWrapper.from_literal(2, "shape")

    def layer_norm2(x, gamma, beta, eps):
        norm = torch.nn.functional.layer_norm(x, [m_shape], gamma, beta, eps)
        return norm

    m_ln2 = layer_norm2(m_a, m_gamma, m_beta, m_eps)
    print(m_ln2)
