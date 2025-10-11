from typing import List, Optional

import torch
import torch.fx as fx
import torch_npu

from xpu_graph.config import OptLevel
from xpu_graph.constant_manager import is_constant
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

ACL_FORMAT_NZ = 29  # NPU optimized NZ format


def npu_format_cast_to_nz(tensor: torch.Tensor) -> torch.Tensor:
    if not hasattr(tensor, "device") or tensor.device.type != "npu":
        return tensor

    try:
        converted_tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT_NZ)
        logger.debug(f"Converted tensor to NZ format: {tensor.shape}")
        return converted_tensor
    except Exception as e:
        logger.warning(f"Failed to convert tensor to NZ format: {e}")
        return tensor


class FoldNdToNzFormat(Pattern):
    _opt_level = OptLevel.level1

    SUPPORTED_OPS_ND2NZ = {
        torch.ops.npu.npu_quant_matmul.default,
    }

    def __init__(self):
        super().__init__()
        self.folding_params = True  # Enable parameter folding for weights

    def _is_constant_weight(self, node: fx.Node) -> bool:
        return is_constant(node, self.folding_params)

    def _is_already_cast_node(self, node: fx.Node) -> bool:
        return node.op == "call_function" and node.target == npu_format_cast_to_nz

    def _is_already_processed(self, op_node: fx.Node) -> bool:
        return op_node.meta.get("nz_format_processed", False)

    def _mark_as_processed(self, op_node: fx.Node):
        op_node.meta["nz_format_processed"] = True

    def _should_process_weight(self, weight_node: fx.Node) -> bool:
        if self._is_already_cast_node(weight_node):
            return False

        return self._is_constant_weight(weight_node)

    def _is_exclusively_used(self, sub_node: fx.Node, node: fx.Node) -> bool:
        return len(sub_node.users) == 1 and list(sub_node.users)[0] == node

    def _insert_format_cast_node(self, gm: fx.GraphModule, weight_node: fx.Node, insert_point: fx.Node) -> fx.Node:
        with gm.graph.inserting_before(insert_point):
            cast_node = gm.graph.create_node(
                op="call_function",
                target=npu_format_cast_to_nz,
                args=(weight_node,),
                name=f"{weight_node.name}_nz_cast",
            )
            cast_node.meta = weight_node.meta.copy()

        logger.debug(f"Inserted NZ format cast for weight: {weight_node.name}")
        return cast_node

    def _process_quant_matmul_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        x2_arg = node.args[1]

        if not self._is_exclusively_used(x2_arg, node):
            logger.warning(
                f"Weight '{x2_arg.name}' is used by multiple nodes "
                f"({len(x2_arg.users)} users). Converting it to NZ format for "
                f"node '{node.name}' might increase memory usage, as the original "
                "tensor will be preserved for other users."
            )

        if self._should_process_weight(x2_arg):
            cast_node = self._insert_format_cast_node(gm, x2_arg, node)

            new_args = list(node.args)
            new_args[1] = cast_node
            node.args = tuple(new_args)

            self._mark_as_processed(node)

            logger.debug("Inserted format cast for x2 weight in npu_quant_matmul")
            return True

        return False

    def process(self, gm: fx.GraphModule) -> bool:
        changed = False
        graph = gm.graph

        candidates = [
            node for node in graph.nodes if node.op == "call_function" and node.target in self.SUPPORTED_OPS_ND2NZ
        ]

        operation_processors = {
            torch.ops.npu.npu_quant_matmul.default: self._process_quant_matmul_weights,
        }

        for node in candidates:
            processor = operation_processors.get(node.target)
            if processor and processor(gm, node):
                changed = True

        return changed
