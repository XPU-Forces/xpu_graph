from typing import Any
import torch
import torch.fx as fx
import re
import os
import inspect
from enum import Enum

from xpu_graph.config import OptLevel
from xpu_graph.passes.optimizer import Optimizer
from xpu_graph.utils import logger


class PatternGroup(Enum):
    GROUP0 = 0
    GROUP1 = 1
    GROUP2 = 2

    def __lt__(self, other):
        if isinstance(other, PatternGroup):
            return self.value < other.value
        return NotImplemented


class Pattern(Optimizer):
    _opt_level = OptLevel.level0
    _pattern_group = PatternGroup.GROUP0

    def __init__(self):
        super().__init__()

    def process(self, gm: fx.GraphModule):
        raise NotImplementedError


class PatternRule(object):
    def __init__(self, type_map: dict, links: dict, end_name: str):
        self._type_map = type_map
        self._links = links
        self._end_name = end_name

    @property
    def type_map(self):
        return self._type_map

    @property
    def links(self):
        return self._links

    @property
    def end_name(self):
        return self._end_name


class AutoMatchPattern(Pattern):
    """
    This method only supports to use positon arguments to define graph now.
    """

    _mermaid_re = re.compile(
        r"^\s*(\w+)([\[\(\{][\w\./]+[\]\)\}]){0,1}\s*-->\s*(\|\d+:[\d,]+\|){0,1}\s*(\w+)([\[\(\{][\w\./]+[\]\)\}]){0,1}\s*$"
    )
    _mermaid_re2 = re.compile(r"^\s*(\w+)([\[\(\{][\w\./]+[\]\)\}]){0,1}\s*$")

    def __init__(self):
        super().__init__()

        self._rule_map = {}

        try:
            with open(self._markdown_path, "r") as file:
                markdown = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Markdown file not found: {self._markdown_path}")

        matchs = re.findall(r"```mermaid\n([\s\S]+?)\n```", markdown)
        if len(matchs) != 1:
            raise RuntimeError("Markdown format error")

        mermaid = matchs[0]

        matchs = re.findall("subgraph (SRC\w*?)\n([\s\S]+?)\nend", mermaid)
        graph_desc_map = {}
        for rule_name, desc in matchs:
            if rule_name not in graph_desc_map:
                graph_desc_map[rule_name] = ""
            graph_desc_map[rule_name] += "\n" + desc

        for rule_name in graph_desc_map:
            self._parse_lines(rule_name, graph_desc_map[rule_name].split("\n"), {}, {})

    @property
    def _markdown_path(self):
        path = inspect.getfile(self.__class__)
        name = os.path.basename(path)
        assert name[-3:] == ".py"
        name = name[:-3] + ".md"
        path = os.path.dirname(path)
        return os.path.join(path, name)

    def _parse_lines(self, rule_name: str, lines: list, type_map: dict, links: dict):
        if not lines:
            idx = 0
            while True:
                if idx == 0:
                    real_rule_name = rule_name
                else:
                    real_rule_name = rule_name + ":" + str(idx)
                idx += 1
                if real_rule_name in self._rule_map:
                    continue
                end_name = None
                names = set([name for name, _ in links.values()])
                for name in type_map.keys():
                    if name not in names:
                        if end_name:
                            raise RuntimeError("Output count of the graph unequal to 1")
                        end_name = name

                if not end_name:
                    raise RuntimeError("Output count of the graph unequal to 1")
                self._rule_map[real_rule_name] = PatternRule(type_map, links, end_name)
                return

        line = lines[0]
        if re.match(r"^\s*$", line):
            return self._parse_lines(
                rule_name, lines[1:], type_map.copy(), links.copy()
            )
        src, dst, types = self._parse_mermaid(line)
        type_map.update(types)
        if not src or not dst:
            return self._parse_lines(
                rule_name, lines[1:], type_map.copy(), links.copy()
            )
        succ = False
        for slot in dst[1]:
            if (dst[0], slot) in links:
                continue
            succ = True
            new_links = links.copy()
            new_links[(dst[0], slot)] = src
            self._parse_lines(rule_name, lines[1:], type_map.copy(), new_links)
        if not succ:
            raise RuntimeError("Topology error")

    def _parse_mermaid(self, line):
        matchs = re.findall(self._mermaid_re, line)
        type_map = {}
        if len(matchs) != 1:
            return self._parse_mermaid_v2(line)
        src, src_type, slots, dst, dst_type = matchs[0]
        if not src or not dst:
            raise RuntimeError("Mermaid format error")
        if src_type:
            src_type = src_type[1:-1]
            type_map[src] = self._get_fx_call_target(src_type)
        if dst_type:
            dst_type = dst_type[1:-1]
            type_map[dst] = self._get_fx_call_target(dst_type)
        if slots:
            src_slot, dst_slot = slots[1:-1].split(":")
            if "," in dst_slot:
                slots = [int(src_slot), [int(s) for s in dst_slot.split(",")]]
            else:
                slots = [int(src_slot), [int(dst_slot)]]
        else:
            slots = [0, [0]]
        return (src, slots[0]), (dst, slots[1]), type_map

    def _parse_mermaid_v2(self, line):
        matchs = re.findall(self._mermaid_re2, line)
        if len(matchs) != 1:
            raise RuntimeError("Mermaid format error")
        src, src_type = matchs[0]
        if not src or not src_type:
            raise RuntimeError("Mermaid format error")
        src_type = src_type[1:-1]
        type_map = {src: self._get_fx_call_target(src_type)}
        return None, None, type_map

    def _get_fx_call_target(self, uris):
        cls_list = []
        for uri in uris.split("/"):
            if not uri:
                continue
            if not (uri.startswith("torch.ops") or uri.startswith("operator")):
                raise RuntimeError(f"Target {uri} is illegal")
            names = uri.split(".")
            import importlib

            value = importlib.import_module(names[0])
            for name in names[1:]:
                if not hasattr(value, name):
                    raise RuntimeError(f"Target[{uri}] not found")
                value = getattr(value, name)

            cls_list.append(value)

        if not cls_list:
            raise RuntimeError("Empty node cls list")
        return tuple(cls_list)

    def process(self, gm: fx.GraphModule):
        for rule_name in self._rule_map:
            if self._process_rule(gm, rule_name):
                return True

        return False

    def _process_rule(self, gm: fx.GraphModule, rule_name: str):
        changed = False
        rule = self._rule_map[rule_name]

        for target in rule.type_map[rule.end_name]:
            candidates = [
                node
                for node in gm.graph.nodes
                if node.op == "call_function" and node.target == target
            ]

            candidates.reverse()
            for cdd in candidates:
                node_map = {}
                node_map[rule.end_name] = cdd
                node_map, matched_rule_set = self._get_match_subgraph(
                    rule, rule.end_name, node_map, set()
                )
                if node_map is None or len(matched_rule_set) != len(rule.links):
                    continue
                if self.rewriter(gm, rule_name, node_map):
                    changed = True

        return changed

    def _get_match_subgraph(
        self, rule: PatternRule, node_alias: str, node_map: dict, matched_rule_set: set
    ):
        for i, parent_node in enumerate(node_map[node_alias].args):
            if not (node_alias, i) in rule.links:
                continue
            if (
                not isinstance(parent_node, fx.Node)
                or parent_node.op != "call_function"
            ):
                return None, set()
            parent_alias, parent_slot = rule.links[(node_alias, i)]

            # Python version: below code is only work for Python3.7+
            import sys

            if sys.version_info >= (3, 7):
                if list(parent_node.users.keys())[parent_slot] != node_map[node_alias]:
                    return None, set()
            else:
                logger.warning("Python version is lower than 3.7")
            if parent_node.target not in rule.type_map[parent_alias]:
                return None, set()

            node_map[parent_alias] = parent_node
            matched_rule_set.add((node_alias, i))

            node_map, matched_rule_set = self._get_match_subgraph(
                rule, parent_alias, node_map, matched_rule_set
            )

        return node_map, matched_rule_set

    def rewriter(self, gm: fx.GraphModule, rule_name: str, node_map: dict) -> bool:
        raise NotImplementedError
