import itertools
import operator
from typing import Callable

import torch
import torch.fx as fx
from torch.fx.node import map_aggregate
from torch.utils._pytree import tree_flatten, tree_iter, tree_map, tree_map_only


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
                self.panic(f"Key {key} already captured with value {self.symbols[key]}, but we got {val} now")
        else:
            self.symbols[key] = val

    def match_val(self, target, val):
        if not self.status:
            return
        if not target == val:
            self.panic(f"Value {val} not match expected {target}")

    def match_node(self, node_capture, node):
        if not self.status:
            return
        if not isinstance(node, fx.Node):
            self.panic(f"Node {node} is not a Fx Node")
            return
        if node_capture.op != node.op:
            self.panic(f"Node op {node.op} not match expected {node_capture.op}")
            return
        if node_capture.target != node.target:
            self.panic(f"Node target {node.target} not match expected {node_capture.target}")
            return

        sub_captures, tree_spec = tree_flatten((node_capture.args, node_capture.kwargs))
        try:
            sub_nodes = tree_spec.flatten_up_to((node.args, node.kwargs))
        except ValueError as e:
            self.panic(
                f"Node argument {node.args} and {node.kwargs} not match expected:\n{node_capture}\nInner error: {e}"
            )
            return
        for sub_capture, sub_node in zip(sub_captures, sub_nodes):
            if not self.status:
                return
            if isinstance(sub_capture, FxCapture):
                if not self.try_capture_node_or_val(sub_capture, sub_node):
                    return
            else:
                self.match_val(sub_capture, sub_node)

    def match_union(self, candidates, node: object):
        state = self.backup_state()
        fail_msgs = []
        for candidate in candidates:
            if self.try_capture_node_or_val(candidate, node):
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
            self.try_capture_node_or_val(candidate, node)

    def try_capture_node_or_val(self, capture, node_or_val: object) -> bool:
        if self.root_capture is None:
            self.root_capture = capture
        if not self.status:
            return False
        if capture.id in self.captured_nodes:
            self.match_val(self.captured_nodes[capture.id], node_or_val)
        elif capture.op == "union":
            self.match_union(capture.args, node_or_val)
        elif capture.op == "product":
            self.match_product(capture.args, node_or_val)
        elif capture.op == "predicate":
            if self.try_capture_node_or_val(capture.args[0], node_or_val):
                if not capture.target(self.symbols):
                    self.panic(
                        f"Meta check for partial captured results failed for already captured:\n"
                        + "\n  ".join(
                            f"{k}: {v.format_node() if isinstance(v, fx.Node) else v}" for k, v in self.symbols.items()
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
                        f"Predicate failed for {node_or_val.format_node() if isinstance(node_or_val, fx.Node) else node_or_val}"
                    )
        elif capture.op in ["call_function", "call_module", "call_method", "get_attr"]:
            self.match_node(capture, node_or_val)
        else:
            raise NotImplementedError(f"Unsupported capture type: {type(capture)}")

        if self.status:
            self.captured_nodes[capture.id] = node_or_val
        return self.status

    def __getitem__(self, id):
        if isinstance(id, str):
            return self.symbols[id]
        elif isinstance(id, FxCapture):
            return self.captured_nodes[id.id]

    def __str__(self):
        if self.root_capture is None:
            return "Unused"
        if self.status:
            s = "Capture success: "
            s += "\n  symbols:"
            for k, v in self.symbols.items():
                s += f"\n    {k}: {v}"
            s += f"\n  captured:"

            def captured_result_repr(cap, cap_id_map, repr_list):
                single_line_repr = cap.single_line_repr(cap_id_map, repr_list)
                if cap.id in self.captured_nodes:
                    captuerd_node = self.captured_nodes[cap.id]
                    captuerd_node_repr = (
                        captuerd_node.format_node() if isinstance(captuerd_node, fx.Node) else captuerd_node
                    )
                    str = f"    {single_line_repr}\n      {captuerd_node_repr}"
                else:
                    str = None
                return str

            repr_list = self.root_capture.to_list(captured_result_repr)
            for line in repr_list:
                if line is not None:
                    s += "\n" + line
            s += "\n"
            return s
        else:
            return f"Capture failed: {self.panic_msg}"


class FxCapture:
    cap_id = itertools.count()

    def __init__(self, op, target, args=None, kwargs=None):
        self.op = op
        self.target = target
        # Keep consistent with node construction
        self.args = map_aggregate(args, lambda x: x) if args is not None else __class__.any()
        self.kwargs = map_aggregate(kwargs, lambda x: x) if kwargs is not None else __class__.any()
        self.id = next(__class__.cap_id)

    def __or__(self, capture):
        return __class__.union(self, capture)

    def __getitem__(self, id):
        return __class__.call_function(operator.getitem, self, id)

    def with_check(self, predicate):
        return __class__.predicate(self, predicate)

    def try_capture(self, node_or_val, ctx=None):
        if ctx is None:
            ctx = CaptureResult()
        result = ctx.try_capture_node_or_val(self, node_or_val)
        return ctx

    def iterate_all_captured(self, nodes, ctx=None):
        if ctx is None:
            ctx = CaptureResult()
        if self.op == "product":
            captures = list(self.args)
        else:
            captures = [self]
        ctx.root_capture = self

        def inner_iterate_all_captured(captures, nodes, ctx, captured_nodes):
            if len(captures) == 0:
                yield ctx, captured_nodes
            else:
                for node in nodes:
                    states = ctx.backup_state()
                    ctx = captures[0].try_capture(node, ctx)
                    if ctx.status:
                        captured_nodes.append(node)
                        yield from inner_iterate_all_captured(captures[1:], nodes, ctx, captured_nodes)
                        captured_nodes.pop()

                    ctx.restore_state(states)

        yield from inner_iterate_all_captured(captures, nodes, ctx, [])

    def replace(self, tracked_literal_to_wrapper):
        def inner_replace(t):
            if t in tracked_literal_to_wrapper:
                t = tracked_literal_to_wrapper[t]
            elif isinstance(t, FxCapture):
                t = t.replace(tracked_literal_to_wrapper)
            return t

        self.args, self.kwargs = tree_map(inner_replace, (self.args, self.kwargs))
        return self

    def single_line_repr(self, node_id_map, result_list):
        if self.op == "symbol":
            if self.target is not None:
                expr = "?" + self.target
            else:
                expr = "_"
            if "predicate" in self.kwargs:
                expr += f" |= [...]"

        elif self.op == "union":
            expr = " || ".join(f"%{node_id_map[cand.id]}" for cand in self.args)
        elif self.op == "product":
            expr = "(" + ", ".join(f"%{node_id_map[cand.id]}" for cand in self.args) + ")"
        elif self.op == "predicate":
            expr = f"meta_check(...)"
        elif self.op in ["call_function", "call_method", "call_module", "get_attr"]:

            class Indexing:
                def __init__(self, idx):
                    self.idx = idx

                def __str__(self):
                    return f"%{self.idx}"

                def __repr__(self):
                    return str(self)

            def map_capture_to_id(obj):
                return Indexing(node_id_map[obj.id])

            expr = f"{self.op}[target={self.target}](args={tree_map_only(FxCapture, map_capture_to_id, self.args)}, kwargs={tree_map_only(FxCapture, map_capture_to_id, self.kwargs)})"
        else:
            raise ValueError(f"Unsupported op: {self.op}")
        return f"%{len(result_list)} := {expr}"

    def to_list(self, map_fn=None, node_id_map=None, result_list=None):
        if node_id_map is None:
            node_id_map = {}
        if result_list is None:
            result_list = []
        if self.id in node_id_map:
            return result_list
        for sub_node in tree_iter((self.args, self.kwargs)):
            if isinstance(sub_node, __class__):
                result_list = sub_node.to_list(map_fn, node_id_map, result_list)
        result = map_fn(self, node_id_map, result_list)
        node_id_map[self.id] = len(result_list)
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
    return FxCapture("call_function", aten._to_copy, args=(capture,)) | FxCapture(
        "call_function", aten._to_copy.default, args=(capture,)
    )


def zero_like():
    return FxCapture("call_function", aten.zeros_like.default) | FxCapture("call_function", aten.zeros.default)


def one_like():
    return FxCapture("call_function", aten.ones_like.default) | FxCapture("call_function", aten.ones.default)


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

    def __str__(self):
        return str(self.matcher) + " | hint=" + str(self.fake_elem)

    def __repr__(self):
        return str(self)


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

    def __repr__(self):
        return str(self)

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
