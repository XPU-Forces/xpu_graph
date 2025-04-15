import logging
import sys
import time
import functools
import os

import torch


class _LoggerWrapper:
    def __init__(self, logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def __setattr__(self, name, value):
        if name == "_logger":
            super().__setattr__(name, value)
        else:
            setattr(self._logger, name, value)


logger = _LoggerWrapper(logging.getLogger("xpu_graph"))


def setup_logger(loglevel):
    if len(logger.handlers) == 0:
        # Skip if handlers already exist
        fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d [XPU_GRAPH][%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(loglevel)


_debug_entries = [
    "xpu_graph." + name for name in os.getenv("XPUGRAPH_LOGS", "").split(",")
]


class local_logger:
    def __init__(self, name, level=None):
        self.name = name
        self.logger = logger.getChild(self.name)

        if self.logger.name in _debug_entries:
            level = logging.DEBUG
        self.level = level or logger.level

    def __enter__(self):
        self.orig_logger = logger._logger
        self.logger.setLevel(self.level)
        logger._logger = self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        logger._logger = self.orig_logger
        self.logger = None
        self.orig_logger = None


def xpu_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()

        # Determine function name (including class name if applicable)
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            func_name = f"{class_name}.{func.__name__}"
        else:
            func_name = func.__name__

        logger.debug(f"{func_name} cost {end - start:.4f}s")
        return res

    return wrapper


def get_nodes_statistics(gm: torch.fx.GraphModule) -> str:
    statistics = []
    statistics.append(f"Total nodes num: {len(gm.graph.nodes)}")
    nodes_map = {
        "placeholder": 0,
        "get_attr": 0,
        "output": 0,
    }
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            nodes_map["placeholder"] += 1
        elif node.op == "get_attr":
            nodes_map["get_attr"] += 1
        elif node.op == "output":
            nodes_map["output"] = len(node.args)
        else:
            callee = node.target
            if callee in nodes_map:
                nodes_map[callee] += 1
            else:
                nodes_map[callee] = 1

    statistics.append(f"{'Node name':<50}  Node num")
    statistics.append("-" * 60)
    statistics += [f"{str(key):<50}  {value}" for key, value in nodes_map.items()]
    statistics_str = "\n".join(statistics)

    return statistics_str
