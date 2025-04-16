import logging
import sys
import time
import functools
import os
import difflib

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


class NodesStatistics:
    def __init__(self):
        self.statistics = {}

    def insert_statistics(self, name: str, gm: torch.fx.GraphModule) -> None:
        # Maybe overwrite, but i don't care
        self.statistics[name] = {
            "total": len(gm.graph.nodes),
            "placeholder": 0,
            "get_attr": 0,
            "output": 0,
        }

        node_map = {}
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                self.statistics[name]["placeholder"] += 1
            elif node.op == "get_attr":
                self.statistics[name]["get_attr"] += 1
            elif node.op == "output":
                self.statistics[name]["output"] = len(node.args)
            else:
                callee = str(node.target)
                if callee in node_map:
                    node_map[callee] += 1
                else:
                    node_map[callee] = 1

        node_map = dict(
            sorted(node_map.items(), key=lambda item: item[1], reverse=True)
        )
        for node_name, cnt in node_map.items():
            self.statistics[name][node_name] = cnt

    def __str__(self):
        node_names = {}
        for node_map in self.statistics.values():
            for name in node_map.keys():
                node_names[name] = None
        node_names = node_names.keys()

        statistics_str = []
        statistics_str.append(f"{'Node name':<50}")
        statistics_str.append("-" * 50)
        for name in self.statistics.keys():
            statistics_str[0] += f"{name:<20}"
            statistics_str[1] += "-" * 20
        for node_name in node_names:
            statistics_str.append(f"{node_name:<50}")

            prev_cnt = None
            for i, nodes_statistics in enumerate(self.statistics.values()):
                node_cnt = (
                    nodes_statistics[node_name] if node_name in nodes_statistics else 0
                )
                cnt_str = f"{node_cnt}"
                if i >= 1:
                    gap = node_cnt - prev_cnt
                    cnt_str += f"({gap})" if gap != 0 else ""
                statistics_str[-1] += f"{cnt_str:<20}"
                prev_cnt = node_cnt

        return "\n" + "\n".join(statistics_str)

class GitLikeDiffer:
    differ = difflib.Differ()

    @classmethod
    def diff(cls, lhs, rhs):
        lhs = str(lhs).splitlines()
        rhs = str(rhs).splitlines()
        diff = cls.differ.compare(lhs, rhs)
        result = []
        is_diff = False
        for line in diff:
            if line.startswith("- "):
                is_diff = True
                # NOTE(liuyuan): Red for removals
                result.append(f"\033[31m{line}\033[0m") 
            elif line.startswith("+ "):
                is_diff = True
                # NOTE(liuyuan): Green for additions
                result.append(f"\033[32m{line}\033[0m") 
            elif line.startswith("? "):
                # NOTE(liuyuan): Yellow for hints
                result.append(f"\033[33m{line.strip()}\033[0m") 
            else:
                # TODO(liuyuan): Is this necessary? Maybe we should ignore it. Maybe.
                result.append(line)
        return '\n'.join(result) if is_diff else "\033[32mNo difference found!\033[0m"