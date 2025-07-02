from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, final

import torch
from torch.fx import Node

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
        res = False
        for self_node in self.nodes:
            res = res or self_node.match(node)
            if res:
                break
        return res

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
    def __init__(self, *args, capture=None, meta_check=None) -> None:
        self.args = args
        for arg in self.args:
            if not isinstance(arg, XpuGraphNode):
                raise ValueError("XpuGraphNode only support XpuGraphNode as input")

        self.op = None
        self.target = set()
        self.capture = capture if isinstance(capture, NodeCapture) else None
        self.meta_check = meta_check if isinstance(meta_check, Callable) else None

    @final
    def match(self, node: Node):
        if self.__match__(node):
            if self.meta_check and not self.meta_check(node):
                return False

            if self.capture:
                self.capture.node = node

            if self.args:
                if len(node.args) != len(self.args):
                    return False

                res = True
                for node_arg, self_arg in zip(node.args, self.args):
                    res = res and self_arg.match(node_arg)
                    if not res:
                        break
                return res
            else:
                return True
        else:
            return False

    def __match__(self, node: Node):
        return (node.op == self.op) and (node.target in self.target)

    def __or__(self, xpu_graph_node):
        assert isinstance(xpu_graph_node, XpuGraphNode)
        cluster = NodeCluster()
        cluster.nodes = [self, xpu_graph_node]
        return cluster


class AnyNode(XpuGraphNode):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    @final
    def __match__(self, node: Node):
        return True


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
class MatMul(CallFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.target.add(aten.mm.default)
        # self.target.add(aten.matmul.default)
        self.target.add(torch.matmul)


if __name__ == "__main__":
    from torch.fx import symbolic_trace

    mm_capture = NodeCapture()

    def pattern():
        # return MatMul(Placeholder(), Placeholder(), capture=mm_capture)
        return MatMul(AnyNode(), AnyNode(), capture=mm_capture)

    def test(x, y):
        return torch.matmul(x, y)

    def diu(x, y):
        return test(test(x, y), test(x, y))

    graph = symbolic_trace(diu).graph

    cnt = 0

    print(graph)
    for node in graph.nodes:
        if pattern().match(node):
            print(mm_capture.node)
            cnt += 1
    assert cnt == 3
