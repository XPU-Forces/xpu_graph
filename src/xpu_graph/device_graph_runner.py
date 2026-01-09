from abc import ABC, abstractmethod
from typing import Callable, ContextManager, Dict, List, Tuple

import torch
import torch.utils._pytree as pytree

from xpu_graph.config import Target
from xpu_graph.utils import ImportOrIgnoreMetaClass, PolyBackendDispatcher


class GraphRunner(torch.nn.Module, ABC, PolyBackendDispatcher):
    def __init__(
        self,
        callable_target: callable,
        init_param_callback: Callable[[Tuple, Dict], List[torch.Tensor]] = None,
        copy_param_callback: Callable[[List[torch.Tensor], Tuple, Dict], bool] = None,
    ):
        super().__init__()
        self._callable_target = callable_target
        self._init_param_callback = init_param_callback
        self._copy_param_callback = copy_param_callback
        self._stream = None
        self._graph = None
        self._input_buffers = None
        self._output = None
        self._mempool = None

    @property
    def mempool(self):
        return self._mempool

    @classmethod
    @abstractmethod
    def _Graph(cls, *args, **kwargs):
        """
        Should return the object of graph class.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _Stream(cls, *args, **kwargs):
        """
        Should return the object of stream class.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _GraphContext(cls, *args, **kwargs) -> ContextManager:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _MempoolHandle(cls):
        raise NotImplementedError

    def _init_param_buffer(self, *args, **kwargs) -> bool:
        """
        init_param_callback should keep the signature the same with the callable target and return the input list of torch.Tensor.
        Example:
            def func(x, y):
                return x + y

            def init_param_callback(x:torch.Tensor, y:torch.Tensor=torch.empty(1024), z=3):
                #WARNING(liuyuan): Never make new one for the torch.Tensor like argument.
                return (x,y)
        """
        if self._init_param_callback:
            self._input_buffers = self._init_param_callback(*args, **kwargs)
            return True
        else:
            return False

    def _copy_to_param_buffer(self, *args, **kwargs) -> bool:
        """
        copy_param_callback should include the signature of the callable target and define how to copy to the tensor list return by init_param_callback
        Example:
            def func(x, y):
                return x + y

            def _copy_param_callbac(input_buffers, x, y):
                try:
                    input_buffers[0].copy_(x)
                    input_buffers[1].copy_(y)
                except Exception:
                    return False
                return True
        """
        if self._copy_param_callback:
            return self._copy_param_callback(self._input_buffers, *args, **kwargs)
        else:
            return False

    def clone(self):
        obj = self.__new__(type(self))
        obj.__init__(
            self._callable_target,
            self._init_param_callback,
            self._copy_param_callback,
        )
        return obj

    def reset(self):
        self.__init__(
            self._callable_target,
            self._init_param_callback,
            self._copy_param_callback,
        )
        return self

    def capture(self, *args, memory_pool=None, clone_args=False, **kwargs) -> None:
        assert self._graph is None, "Device graph has been recorded."
        self._stream = self._Stream()
        self._graph = self._Graph()
        self._mempool = memory_pool if memory_pool else self._MempoolHandle()

        if clone_args:
            args, kwargs = pytree.tree_map(
                lambda x: (x.detach().clone() if isinstance(x, torch.Tensor) else x),
                (args, kwargs),
            )

        # NOTE(liuyuan): just hold the input tensors.
        assert self._init_param_buffer(*args, **kwargs)
        assert self._input_buffers is not None

        with self._GraphContext(
            self._graph,
            stream=self._stream,
            pool=self._mempool,
        ):
            self._output = self._callable_target(
                *args,
                **kwargs,
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        assert self._copy_to_param_buffer(*args, **kwargs)
        self._graph.replay()
        self._stream.synchronize()
        return self._output


# NOTE(liuyuan): Thx to the torch device mechanism, all we need is to import the extention.
# And we create or set the class to None according to the result of importing the extention.
class NPUGraphRunner(
    GraphRunner,
    backend=Target.npu,
    anchor_class=GraphRunner,
    metaclass=ImportOrIgnoreMetaClass,
    __modules_to_import__="torch_npu",
):
    @classmethod
    def _Graph(cls, *args, **kwargs):
        return torch.npu.NPUGraph(*args, **kwargs)

    @classmethod
    def _Stream(cls, *args, **kwargs):
        return torch.npu.Stream(*args, **kwargs)

    @classmethod
    def _GraphContext(cls, *args, **kwargs):
        return torch.npu.graph(*args, **kwargs)

    @classmethod
    def _MempoolHandle(cls):
        return torch.npu.graph_pool_handle()


class MLUGraphRunner(
    GraphRunner,
    backend=Target.mlu,
    anchor_class=GraphRunner,
    metaclass=ImportOrIgnoreMetaClass,
    __modules_to_import__="torch_mlu",
):
    @classmethod
    def _Graph(cls, *args, **kwargs):
        return torch.mlu.MLUGraph(*args, **kwargs)

    @classmethod
    def _Stream(cls, *args, **kwargs):
        return torch.mlu.Stream(*args, **kwargs)

    @classmethod
    def _GraphContext(cls, *args, **kwargs):
        return torch.mlu.graph(*args, **kwargs)

    @classmethod
    def _MempoolHandle(cls):
        return torch.mlu.graph_pool_handle()
