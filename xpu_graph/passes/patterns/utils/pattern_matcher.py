import operator
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Optional, final

import torch
from torch.fx import Node as FxNode

aten = torch.ops.aten
from torch.utils._pytree import tree_iter, tree_map

from xpu_graph.utils import logger


def xnary(operand_num: int):
    """check that the number of args MUST be equal to operand_num
       before calling __init__ of the class.

    Args:
        operand_num (int): the number of operands.
    """

    def decorate(cls):
        func = cls.__init__

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            assert len(args) == operand_num, f"Expected {operand_num} operand(s) but we got {len(args)} now"
            return func(self, *args, **kwargs)

        cls.__init__ = wrapper
        return cls

    return decorate


class CaptureCtx:
    def __init__(self):
        self.status = True
        self.captured = {}

    def __repr__(self):
        return str(self.captured) if self.status else "!panic"

    def match_key(self, key, val) -> bool:
        if not self.status:
            return False
        assert key is not None
        if key in self.captured:
            if not self.captured[key] == val:
                self.panic()
        else:
            self.captured[key] = val
        return self.status

    def match_val(self, target, val) -> bool:
        if not self.status:
            return False
        if not isinstance(target, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            if not target == val:
                self.panic()
        else:
            key = target.node.expr
            if key in self.captured:
                if not self.captured[key] == val:
                    self.panic()
            else:
                self.captured[key] = val
        return self.status

    def update(self, flag):
        if not self.status:
            return False
        if not flag:
            self.panic()
        return self.status

    def panic(self, msg=""):
        self.status = False
        return False

    def backup_state(self):
        return (self.status, set(self.captured.keys()))

    def restore_state(self, state):
        self.status, captured_keys = state
        self.captured = {k: self.captured[k] for k in captured_keys}


class BaseCapture(ABC):
    @abstractmethod
    def match(self, ctx: CaptureCtx, node_or_val: object) -> bool:
        raise NotImplementedError()

    def __or__(self, capture):
        return UnionCapture(self, capture)

    def __and__(self, capture):
        return UnionCapture(self, capture)

    def with_check(self, predicate):
        return PredicateCapture(self, predicate)

    def __call__(self, ctx, node_or_val):
        # print(f"matching: {self}")
        # print(f"target: {node.format_node() if isinstance(node, FxNode) else node}")
        result = self.match(ctx, node_or_val)
        # print(f"result: {ctx}")
        # print()
        return result


class UnionCapture(BaseCapture):
    def __init__(self, *candidates):
        self.candidates = []
        for candidate in candidates:
            if isinstance(candidate, UnionCapture):
                self.candidates.extend(candidate.candidates)
            else:
                self.candidates.append(candidate)

    @final
    def match(self, ctx: CaptureCtx, node: object) -> bool:
        state = ctx.backup_state()
        for candidate in self.candidates:
            if candidate(ctx, node):
                return True
            else:
                ctx.restore_state(state)
        return ctx.panic()

    def __repr__(self):
        return " | ".join([f"{cand}" for cand in self.candidates])


class PredicateCapture(BaseCapture):
    def __init__(
        self,
        capture: BaseCapture,
        meta_check: Callable[[dict], bool],
    ) -> None:
        self.capture = capture
        self.meta_check = meta_check

    @final
    def match(self, ctx: CaptureCtx, node_or_val: object) -> bool:
        if self.capture(ctx, node_or_val):
            return ctx.update(self.meta_check(ctx.captured))
        return False

    def __repr__(self):
        return f"{self.capture}?[...]"


class AnyCapture(BaseCapture):
    def __init__(self, predicate: Optional[Callable[..., bool]] = None):
        self.predicate = predicate

    @final
    def match(self, ctx: CaptureCtx, node_or_val: object) -> bool:
        if self.predicate is None:
            return ctx.update(True)
        return ctx.update(self.predicate(node_or_val))

    def __repr__(self):
        return "_" if self.predicate is None else f"_?[{self.predicate}]"


class PlaceholderCapture(BaseCapture):
    def __init__(self, symbol: str):
        self.symbol = symbol

    @final
    def match(self, ctx: CaptureCtx, node_or_val: object) -> bool:
        return ctx.match_key(self.symbol, node_or_val)

    def __repr__(self):
        return "?" + self.symbol


class NodeCapture(BaseCapture):
    def __init__(self, op, target, args, kwargs):
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs

    @final
    def match(self, ctx: CaptureCtx, node_or_val: object) -> bool:
        if not isinstance(node_or_val, FxNode):
            ctx.panic()
            return ctx

        node = node_or_val

        ctx.match_val(self.op, node.op)
        ctx.match_val(self.target, node.target)
        for sub_capture, sub_node in zip(tree_iter((self.args, self.kwargs)), tree_iter((node.args, node.kwargs))):
            if isinstance(sub_capture, BaseCapture):
                if not sub_capture(ctx, sub_node):
                    return False
            else:
                ctx.match_val(sub_capture, sub_node)

        return ctx.status

    def __repr__(self):
        return f"{self.op}[target={self.target}](args={self.args}, kwargs={self.kwargs})"


class CaptureBuilder:
    @staticmethod
    def placeholder(symbol: str) -> BaseCapture:
        return PlaceholderCapture(symbol)

    @staticmethod
    def any() -> BaseCapture:
        return AnyCapture()

    @staticmethod
    def any_if(predicate: Callable[..., bool]) -> BaseCapture:
        return AnyCapture(predicate)

    @staticmethod
    def call_function(target, *args, **kwargs) -> BaseCapture:
        return NodeCapture("call_function", target, args, kwargs)

    @staticmethod
    def call_method(target, *args, **kwargs) -> BaseCapture:
        return NodeCapture("call_method", target, args, kwargs)

    @staticmethod
    def call_module(target, *args, **kwargs) -> BaseCapture:
        return NodeCapture("call_module", target, args, kwargs)

    @staticmethod
    def get_attr(target) -> BaseCapture:
        return NodeCapture("get_attr", target, (), {})


def matmul_like(m_capture, n_capture):
    if m_capture is None:
        m_capture = CaptureBuilder.any()
    if n_capture is None:
        n_capture = CaptureBuilder.any()
    return CaptureBuilder.call_function(aten.mm.default, m_capture, n_capture) | CaptureBuilder.call_function(
        aten.matmul.default, m_capture, n_capture
    )


def add_like(left_capture, right_capture):
    if left_capture is None:
        left_capture = CaptureBuilder.any()
    if right_capture is None:
        right_capture = CaptureBuilder.any()
    return CaptureBuilder.call_function(aten.add.Scalar, left_capture, right_capture) | CaptureBuilder.call_function(
        aten.add.Tensor, left_capture, right_capture
    )


def dtype_cast_like(capture):
    if capture is None:
        capture = CaptureBuilder.any()
    return CaptureBuilder.call_function(aten._to_copy, capture) | CaptureBuilder.call_function(
        aten._to_copy.default, capture
    )


def zero_like():
    return CaptureBuilder.call_function(aten.zeros_like.default, CaptureBuilder.any()) | CaptureBuilder.call_function(
        aten.zeros.default
    )


def one_like():
    return CaptureBuilder.call_function(aten.ones_like.default, CaptureBuilder.any()) | CaptureBuilder.call_function(
        aten.ones.default
    )


class LiteralCaptureWrapper:
    def __init__(self, fake_elem, mathcer):
        self.fake_elem = fake_elem
        self.matcher = mathcer

    @staticmethod
    def from_literal(literal, symbol):
        return LiteralCaptureWrapper(literal, CaptureBuilder.placeholder(symbol))


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
            elif isinstance(t, LiteralCaptureWrapper):
                return t.fake_elem
            else:
                return t

        def get_matcher_fn(t):
            if isinstance(t, cls):
                return t.matcher
            elif isinstance(t, torch.Tensor):
                return CaptureBuilder.any()
            elif isinstance(t, LiteralCaptureWrapper):
                return t.matcher
            else:
                return t

        kwargs = kwargs or {}

        r_matcher = CaptureBuilder.call_function(func, tree_map(get_matcher_fn, args), tree_map(get_matcher_fn, kwargs))
        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def gen_getitem_matcher(src, fake_elem, i):
            if isinstance(fake_elem, torch.Tensor):
                return cls(fake_elem, CaptureBuilder.call_function(operator.getitem, (src, i)))
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

    def __repr__(self):
        return str(self.matcher)

    @staticmethod
    def from_tensor(tensor, name):
        return __class__(tensor, CaptureBuilder.placeholder(name))

    def match_node(self, node):
        ctx = CaptureCtx()
        self.matcher(ctx, node)
        return ctx


if __name__ == "__main__":
    import logging

    from torch.fx.experimental.proxy_tensor import make_fx

    from xpu_graph.utils import setup_logger

    setup_logger(logging.DEBUG)

    def check_dtype(node):
        if not isinstance(node, FxNode) or node.meta["tensor_meta"].dtype == torch.float32:
            return False
        return True

    def pattern():
        m_input = CaptureBuilder.placeholder("input").with_check(
            lambda captured: "input" in captured and check_dtype(captured["input"])
        )
        m_weight = CaptureBuilder.any_if(check_dtype)
        return matmul_like(m_input | dtype_cast_like(m_input), m_weight | dtype_cast_like(m_weight))

    def foo(x, y):
        return torch.matmul(torch.matmul(x, y).to(torch.int32), torch.matmul(x, y).to(torch.int32))

    graph = make_fx(foo)(torch.empty(1, 1024), torch.empty(1024, 1)).graph

    cnt = 0

    print(graph)
    pat = pattern()
    for node in reversed(graph.nodes):
        ctx = CaptureCtx()
        if pat(ctx, node):
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
            CaptureBuilder.any(),
            CaptureBuilder.placeholder("bias1"),
        ),
        add_like(
            CaptureBuilder.any(),
            CaptureBuilder.placeholder("bias2"),
        ),
    )
    for node in reversed(graph.nodes):
        ctx = CaptureCtx()
        if pattern(ctx, node):
            print(ctx)
            cnt += 1
    assert cnt == 1
    print(cnt)
