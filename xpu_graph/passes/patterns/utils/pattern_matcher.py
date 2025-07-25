import inspect
from abc import ABC, abstractmethod
from functools import cache, wraps
from typing import Callable, final

import torch
from torch.fx import Node
from torch.ops import aten

from xpu_graph.utils import logger

# from torch.ops import aten


def xnary(operand_num):
    def decorate(cls):
        func = cls.__init__

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            assert len(args) == operand_num, f"Expected {operand_num} operand(s) but we got {len(args)} now"
            return func(self, *args, **kwargs)

        cls.__init__ = wrapper
        return cls

    return decorate


class BaseNode(ABC):
    @abstractmethod
    def match(self, node: Node) -> bool:
        pass


class NodeCluster(BaseNode):
    def __init__(self):
        self.nodes = []

    @final
    def match(self, node: Node):
        for self_node in self.nodes:
            if self_node.match(node):
                return True
        return False

    def __or__(self, xpu_graph_node):
        self.nodes.append(xpu_graph_node)
        return self


class NodeCapture:
    def __init__(self):
        self.__node = None

    @property
    def node(self):
        return self.__node

    @node.setter
    def node(self, node: Node):
        self.__node = node

    def clear(self):
        self.__node = None


class XpuGraphNode(BaseNode):
    def __init__(
        self,
        *args,
        capture: NodeCapture = None,
        meta_check: Callable[[dict], bool] = None,
    ) -> None:
        self.args = args
        for arg in self.args:
            if not isinstance(arg, XpuGraphNode) and not isinstance(arg, NodeCluster):
                raise TypeError("XpuGraphNode only support [XpuGraphNode, NodeCluster] as input")

        self.op = None
        self.target = set()
        self.capture = capture if isinstance(capture, NodeCapture) else None
        self.meta_check = meta_check if isinstance(meta_check, Callable) else None

    @final
    def match(self, node: Node):
        if not self.__match__(node):
            logger.lzdbg(
                lambda: f"{self} failed to match "
                + (f"{node.op}={node.target}" if isinstance(node, Node) else f"{node}")
                + (
                    f" with different argument number, expected:{len(self.args)}, actual:{len(node.args)}"
                    if self.args and hasattr(node, "args") and len(self.args) != len(node.args)
                    else ""
                )
            )
            return False

        if self.meta_check and not self.meta_check(node.meta):
            logger.lzdbg(lambda: f"{self} failed in meta_check:\n{inspect.getsource(self.meta_check)}")
            return False

        logger.lzdbg(
            lambda: (f"{self} matched " + (f"{node.op}={node.target}" if isinstance(node, Node) else f"{node}"))
        )
        if self.capture:
            self.capture.node = node

        if self.args and hasattr(node, "args"):
            for node_arg, self_arg in zip(node.args, self.args):
                if not self_arg.match(node_arg):
                    return False

        return True

    def __match__(self, node: Node):
        return (
            (node.op == self.op) and (node.target in self.target) and (len(node.args) == len(self.args))
            if isinstance(node, Node)
            else False
        )

    def __or__(self, xpu_graph_node):
        assert isinstance(xpu_graph_node, XpuGraphNode)
        cluster = NodeCluster()
        cluster.nodes = [self, xpu_graph_node]
        return cluster

    @cache
    def __str__(self):
        return f"XpuGrapnNode: {self.op}={[str(t) for t in self.target]}"


class AnyNode(XpuGraphNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.op = "Any"

    @final
    def __match__(self, node: Node):
        return True


class LiteralNode(XpuGraphNode):
    def __init__(self, literal, *args, ignore_val=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.op = type(literal)
        self.target.add(literal)
        self.ignore_val = ignore_val

    @final
    def __match__(self, node: Node):
        if isinstance(node, self.op):
            return self.ignore_val or node in self.target
        else:
            return False


@xnary(0)
class Placeholder(XpuGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = "placeholder"

    def __match__(self, node):
        return node.op == self.op


class CallFunction(XpuGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = "call_function"


class CallMethod(XpuGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = "call_method"


class CallModule(XpuGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = "call_module"


class GetAttr(XpuGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = "get_attr"


@xnary(2)
class AtenMatMul(CallFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target.update((aten.mm.default, aten.matmul.default))


@xnary(2)
class AtenAdd(CallFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target.update((aten.add.Scalar, aten.add.Tensor))


@xnary(1)
class AtenDTypeCast(CallFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target.update((aten._to_copy, aten._to_copy.default))


@xnary(1)
class AtenZeroLike(CallFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target.update((aten.zeros_like.default, aten.zeros.default))


if __name__ == "__main__":
    import logging

    from torch.fx.experimental.proxy_tensor import make_fx

    from xpu_graph.utils import setup_logger

    setup_logger(logging.DEBUG)

    mm_capture = NodeCapture()

    def pattern():
        def checkMatmul(meta):
            if meta["tensor_meta"].dtype == torch.float32:
                return False
            return True

        return AtenMatMul(
            Placeholder() | AtenDTypeCast(AnyNode()),
            Placeholder() | AtenDTypeCast(AnyNode()),
            capture=mm_capture,
            meta_check=checkMatmul,
        )

    def diu(x, y):
        return torch.matmul(torch.matmul(x, y).to(torch.int32), torch.matmul(x, y).to(torch.int32))

    graph = make_fx(diu)(torch.empty(1, 1024), torch.empty(1024, 1)).graph

    cnt = 0

    print(graph)
    for node in reversed(graph.nodes):
        if pattern().match(node):
            print(mm_capture.node.meta)
            cnt += 1
    print(cnt)
    assert cnt == 1

    def test(x, y):
        return torch.matmul(x + 2, y + 2)

    graph = make_fx(test)(torch.empty(1, 1024), torch.empty(1024, 1)).graph
    print(graph)
    cnt = 0
    pattern = AtenMatMul(
        AtenAdd(Placeholder(), AtenAdd(AnyNode(), AnyNode())),
        AtenAdd(Placeholder(), AtenAdd(AnyNode(), AnyNode())),
    )
    for node in reversed(graph.nodes):
        if pattern.match(node):
            print(mm_capture.node.meta)
            cnt += 1
    print(cnt)
