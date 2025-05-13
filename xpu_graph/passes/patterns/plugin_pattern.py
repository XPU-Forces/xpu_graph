import inspect
from contextlib import contextmanager
from typing import AnyStr, Callable, List, Union

from torch import fx
from torch.fx import subgraph_rewriter, symbolic_trace

from xpu_graph.config import Target
from xpu_graph.fx_utils import FxStage, dispatch_graph
from xpu_graph.utils import logger

from .pattern import Pattern

__all__ = [
    "register_plugin_pattern",
    "register_this_as_plugin_pattern",
    "register_this_as_pattern_constraint",
    "deregister_plugin_patterns",
    "enable_plugin_patterns",
]


class PluginPattern(Pattern):
    def __init__(
        self,
        eager_func: Callable,
        example_inputs,
        replacement: Callable,
        filter_list: List[Callable[["InternalMatch", fx.GraphModule, fx.GraphModule], bool]] = (lambda: True,),
        is_training=False,
        name=None,
    ):
        super().__init__()
        super().set_current_stage(FxStage.pregrad if is_training else FxStage.inference)
        self._eager_pattern, _ = dispatch_graph(symbolic_trace(eager_func), example_inputs, stage=self._current_stage)
        logger.debug("Pattern %s-%s : %s", eager_func.__module__, eager_func.__name__, self._eager_pattern.graph)
        self._replacement = replacement
        self._filter_list = list(filter_list) if filter_list is not None else []
        self.name = name

    @property
    def filter_list(self):
        return self._filter_list

    def process(self, gm: fx.GraphModule):
        matched = subgraph_rewriter.replace_pattern_with_filters(
            gm, self._eager_pattern, self._replacement, self._filter_list, ignore_literals=True
        )
        return len(matched) > 0

    def __str__(self):
        if self.name:
            return self.name
        else:
            return super().__str__()


__PLUGIN_PATTERN_GROUP__ = {}
__PLUGIN_PATTERN_FILTER_FUNCS__: dict = {}
__LAST_PATTERN_INFO__ = None
__CONTEXTUAL_PATTERN_RECORDER__ = None


def register_plugin_pattern(
    eager_func: Callable,
    example_inputs,
    replacement: Callable,
    target: Target,
    extra_checks=None,
    is_training=False,
    postfix=None,
):
    func_name = f"{eager_func.__module__}-{eager_func.__name__}"
    sign = inspect.signature(eager_func)

    assert len(example_inputs) == len(
        sign.parameters
    ), f"You should provide the same number of example inputs as the number of parameters in the function {func_name}"
    assert inspect.signature(replacement) == sign, (
        f"The signature of the replacement function {replacement.__name__} "
        f"should be the same as the signature of the function {func_name}."
    )

    def transform_check_funcs(func):
        def param_check(internalmatch, gm, pg):
            placeholder_tensors = []
            for p in internalmatch.placeholder_nodes:
                placeholder_tensors.append(p.meta["val"])
            result = func(*placeholder_tensors)
            assert isinstance(result, bool)
            return result

        return param_check

    if extra_checks:
        extra_checks = [transform_check_funcs(func) for func in extra_checks]

    global __CONTEXTUAL_PATTERN_RECORDER__
    if __CONTEXTUAL_PATTERN_RECORDER__ is not None and isinstance(__CONTEXTUAL_PATTERN_RECORDER__, list):
        __CONTEXTUAL_PATTERN_RECORDER__.append(eager_func)

    __PLUGIN_PATTERN_GROUP__.setdefault(target, {}).setdefault(func_name, []).append(
        PluginPattern(
            eager_func,
            example_inputs,
            replacement,
            extra_checks,
            is_training,
            "-".join((func_name, postfix)) if postfix else func_name,
        )
    )

    global __LAST_PATTERN_INFO__
    __LAST_PATTERN_INFO__ = (func_name, sign, target)


def register_this_as_plugin_pattern(
    example_inputs,
    replacement,
    target,
    extra_checks=None,
    is_training=False,
    argument_elimination=None,
    postfix=None,
):
    def new_func_maker(func):
        __META_FUNC_STR__ = "def new_func{sign}: new_args, new_kwargs = {args_generator}{sign};return {original_func}(*new_args, **new_kwargs)"
        sign = inspect.signature(argument_elimination)
        assert len(sign.parameters) <= len(inspect.signature(func).parameters), (
            f"The number of parameters in the argument elimination function {argument_elimination.__name__} "
            f"should be less than or equal to the number of parameters in the function {func.__name__}."
        )
        local_vars = {}
        global_vars = {"argument_elimination": argument_elimination, "func": func}
        # NOTE(liuyuan): Define different function with different signature dynamically, because symbolic_trace does not support variadic fucntion.
        exec(
            __META_FUNC_STR__.format(sign=sign, args_generator="argument_elimination", original_func="func"),
            global_vars,
            local_vars,
        )
        new_func = local_vars["new_func"]
        new_func.__name__ = func.__name__
        new_func.__module__ = func.__module__
        new_func.__signature__ = sign
        return new_func

    def decorate(func):
        if argument_elimination:
            # NOTE(liuyuan): do not replace the func that we're going to return as follows.
            # func = new_func_maker(func)
            # replacement = new_func_maker(replacement)
            register_plugin_pattern(
                new_func_maker(func),
                example_inputs,
                new_func_maker(replacement),
                target,
                extra_checks,
                is_training,
                postfix,
            )
        else:
            register_plugin_pattern(
                func,
                example_inputs,
                replacement,
                target,
                extra_checks,
                is_training,
                postfix,
            )
        # NOTE(liuyuan): No need to wrap the function to be decorated because we are not going to change the invoking to it.
        return func

    return decorate


def register_this_as_pattern_constraint(func):
    global __LAST_PATTERN_INFO__
    assert __LAST_PATTERN_INFO__ is not None
    func_name, signature, target = __LAST_PATTERN_INFO__
    assert inspect.signature(func) == signature
    pattern_list = __PLUGIN_PATTERN_GROUP__.get(target).get(func_name)

    def param_check(internalmatch, gm, pg):
        placeholder_tensors = []
        for p in internalmatch.placeholder_nodes:
            placeholder_tensors.append(p.meta["val"])
        result = func(*placeholder_tensors)
        assert isinstance(result, bool)
        return result

    for pattern in pattern_list:
        pattern.filter_list.append(param_check)
    logger.debug("Register constraint %s for %s successfully.", func.__name__, func_name)
    return func


# WARNING(liuyuan): MUST be Union instead of '|' because npu-ci is under python3.9 which does not support '|' in type hint.
def deregister_plugin_patterns(func_or_func_name: Union[Callable, AnyStr], target=None):
    if isinstance(func_or_func_name, Callable):
        func_name = f"{func_or_func_name.__module__}-{func_or_func_name.__name__}"
    elif isinstance(func_or_func_name, str):
        func_name = func_or_func_name
    else:
        raise ValueError("func_or_func_name should be Callable or str")

    if target:
        if func_name in __PLUGIN_PATTERN_GROUP__.get(target, {}):
            del __PLUGIN_PATTERN_GROUP__[target][func_name]
            logger.debug("Deregister pattern %s successfully.", func_name)
    else:
        for target_patterns in __PLUGIN_PATTERN_GROUP__.values():
            if func_name in target_patterns:
                del target_patterns[func_name]
                logger.debug("Deregister pattern %s successfully.", func_name)


@contextmanager
# WARNING(liuyuan): NOT Thread safe but GIL does not care.
def enable_plugin_patterns():
    global __CONTEXTUAL_PATTERN_RECORDER__
    old_recorder = __CONTEXTUAL_PATTERN_RECORDER__
    __CONTEXTUAL_PATTERN_RECORDER__ = []
    try:
        yield
    finally:
        for func in __CONTEXTUAL_PATTERN_RECORDER__:
            deregister_plugin_patterns(func)
        __CONTEXTUAL_PATTERN_RECORDER__ = old_recorder
