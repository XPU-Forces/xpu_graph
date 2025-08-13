import inspect
from abc import ABC, ABCMeta, abstractmethod
from functools import cache, wraps
from typing import Callable, final

import torch
from torch.fx import Node as FxNode
from torch.ops import aten

from xpu_graph.utils import logger

def xnary(operand_num:int):
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


class BaseNode(ABC):
    @abstractmethod
    def match(self, node: FxNode) -> bool:
        pass

class NodeCluster(BaseNode):
    def __init__(self):
        self.nodes = []

    @final
    def match(self, node: FxNode):
        """Whether the node matches ANY one in cluster or not.

        Args:
            node (FxNode): fx.Graph Node

        Returns:
            bool
        """
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
    def node(self, node: FxNode):
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
        """A BaseNode that provide matching function.

        Args:
            ...: The operands of the node.
            capture (NodeCapture, optional): If capture is provided, then the fx.Graph node that matched will be recorded.
                                             Defaults to None.
            meta_check (Callable[[dict], bool], optional): If meta_check is provided, then the fx.Graph node's meta infomation will be checked while matching.
                                                           Defaults to None.

        Raises:
            TypeError: The operands must be XpuGraphNode or NodeCluster. 
        """
        self.args = args
        for arg in self.args:
            if not isinstance(arg, XpuGraphNode) and not isinstance(arg, NodeCluster):
                breakpoint()
                raise TypeError("XpuGraphNode only support [XpuGraphNode, NodeCluster] as input")

        self.op = None
        self.target = set()

        # NOTE(liuyuan): Would be deprecated one day. Setting meta information in __init__ will not supported.
        if not hasattr(self, 'capture'):
            self.capture = capture if isinstance(capture, NodeCapture) else None

        if not hasattr(self, 'meta_check'):
            self.meta_check = meta_check if isinstance(meta_check, Callable) else None

    @final
    def __call__(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        return self

    def __set_meta__(self, capture=None, meta_check=None):
        self.capture = capture if isinstance(capture, NodeCapture) else None
        self.meta_check = (
            meta_check if isinstance(meta_check, Callable) else None
        )

    @final
    def __class_getitem__(cls, key):
        node = super().__new__(cls)
        if isinstance(key, tuple):
            node.__set_meta__(*key)
        elif isinstance(key, dict):
            node.__set_meta__(**key)
        else:
            node.__set_meta__(key)
        return node

    @final
    def match(self, node: FxNode):
        """Whether fx.Graph Node is matched or not.

        Args:
            node (FxNode): fx.Graph Node to match.

        Returns:
            bool
        """
        if not self.__match__(node):
            logger.lzdbg(
                lambda: f"{self} failed to match "
                + (f"{node.op}={node.target}" if isinstance(node, FxNode) else f"{node}")
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
            lambda: (f"{self} matched " + (f"{node.op}={node.target}" if isinstance(node, FxNode) else f"{node}"))
        )

        if self.args and hasattr(node, "args"):
            for node_arg, self_arg in zip(node.args, self.args):
                if not self_arg.match(node_arg):
                    return False

        # NOTE(liuyuan): SHOULD capture the fx.Graph node iff all operands are matched.
        if self.capture:
            self.capture.node = node

        return True

    def __match__(self, node: FxNode):
        return (
            (node.op == self.op) and (node.target in self.target) and (len(node.args) == len(self.args))
            if isinstance(node, FxNode)
            else False
        )

    def __or__(self, other):
        """Return a NodeCluster including self and other

        Args:
            other (XpuGraphNode): The other XpuGraphNode

        Returns:
            cluster (NodeCluster): a NodeCluster including self and other
        """
        assert isinstance(other, XpuGraphNode)
        cluster = NodeCluster()
        cluster.nodes = [self, other]
        return cluster

    @cache
    def __str__(self):
        return f"XpuGrapnNode: {self.op}={[str(t) for t in self.target]}"


class AnyNode(XpuGraphNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.op = "Any"

    @final
    def __match__(self, node: FxNode):
        return True


@xnary(1)
class LiteralNode(XpuGraphNode):
    def __init__(self, literal, ignore_val=False, **kwargs) -> None:
        """A XpuGraphNode that indicating a literal val.

        Args:
            literal (ScalarType): a scalar value. Could be literal.
            ignore_val (bool, optional): whether to ignore the match of val or not. Defaults to False.
        """
        super().__init__(**kwargs)
        self.op = type(literal)
        self.target.add(literal)
        if not hasattr(self, 'ignore_val'):
            self.ignore_val = ignore_val

    @final
    def __match__(self, node: FxNode):
        if isinstance(node, self.op):
            return self.ignore_val or node in self.target
        else:
            return False

    def __set_meta__(self, ignore_val: bool = False, **kwargs):
        self.ignore_val = ignore_val
        super().__set_meta__(**kwargs)

    def __str__(self):
        return super().__str__() + f"{{ignore_val={self.ignore_val}}}"


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

    def checkMatmul(meta):
        if meta["tensor_meta"].dtype == torch.float32:
            return False
        return True

    def pattern():
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
    literal_cap = NodeCapture()
    add_cap = NodeCapture()
    mm_capture.clear()
    pattern = AtenMatMul(
        AtenAdd(
            Placeholder(),
            LiteralNode[dict(ignore_val=True, capture=literal_cap)](100),
        ),
        AtenAdd(
            Placeholder(), AtenAdd[add_cap, checkMatmul](AnyNode(), AnyNode())
        ),
        capture=mm_capture,
    )
    for node in reversed(graph.nodes):
        if pattern.match(node):
            print(mm_capture.node.meta)
            print(literal_cap.node)
            cnt += 1
    assert literal_cap.node 
    assert add_cap.node is None and mm_capture.node is None
    print(cnt)
