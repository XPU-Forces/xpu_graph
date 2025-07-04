from typing import List, Optional

import torch
import torch.fx as fx

from xpu_graph.config import OptLevel
from xpu_graph.constant_manager import is_constant
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger

import torch_npu
ACL_FORMAT_NZ = 29  # NPU optimized NZ format


def npu_format_cast_to_nz(tensor: torch.Tensor) -> torch.Tensor:
    if not hasattr(tensor, "device") or tensor.device.type != "npu":
        return tensor

    try:
        converted_tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT_NZ)
        logger.info(f"Converted tensor to NZ format: {tensor.shape}")
        return converted_tensor
    except Exception as e:
        logger.warning(f"Failed to convert tensor to NZ format: {e}")
        return tensor


class FoldNdToNzFormat(Pattern):
    _opt_level = OptLevel.level1

    SUPPORTED_OPS_ND2NZ = {
        torch.ops.npu.npu_quant_matmul.default,
        torch.ops.aten.mm.default,
        # NOTE (@zhaowenshuo 6.26) As of now, torch NPU only supports quant matmul operators with NZ format weights, so the following operators are temporarily on hold.
        # torch.ops.npu.npu_grouped_matmul.default,
        # torch.ops.npu.npu_grouped_matmul_finalize_routing.default,
        # torch.ops.npu.npu_weight_quant_batchmatmul.default,
        # torch.ops.npu.npu_mla_prolog.default,
        # torch.ops.npu.npu_convert_weight_to_int4pack.default,
        # torch.ops.npu.npu_mm_all_reduce_base.default,
        # torch_npu.npu_grouped_matmul_finalize_routing,
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

    def _insert_format_cast_node(self, gm: fx.GraphModule, weight_node: fx.Node, insert_point: fx.Node) -> fx.Node:
        with gm.graph.inserting_before(insert_point):
            cast_node = gm.graph.create_node(
                op="call_function",
                target=npu_format_cast_to_nz,
                args=(weight_node,),
                name=f"{weight_node.name}_nz_cast",
            )
            cast_node.meta = weight_node.meta.copy()

        logger.info(f"Inserted NZ format cast for weight: {weight_node.name}")
        return cast_node

    def _process_quant_matmul_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        x2_arg = node.args[1]

        if self._should_process_weight(x2_arg):
            cast_node = self._insert_format_cast_node(gm, x2_arg, node)

            new_args = list(node.args)
            new_args[1] = cast_node
            node.args = tuple(new_args)

            self._mark_as_processed(node)

            logger.info("Inserted format cast for x2 weight in npu_quant_matmul")
            return True

        return False

    def _process_grouped_matmul_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        weight_arg = node.args[1]
        changed = False

        if isinstance(weight_arg, list):
            new_weights = []

            for weight_node in weight_arg:
                if self._should_process_weight(weight_node):
                    cast_node = self._insert_format_cast_node(gm, weight_node, node)
                    new_weights.append(cast_node)
                    changed = True
                    logger.info(f"Inserted format cast for weight in grouped_matmul: {weight_node.name}")
                else:
                    new_weights.append(weight_node)

            if changed:
                new_args = list(node.args)
                new_args[1] = new_weights
                node.args = tuple(new_args)

        elif self._should_process_weight(weight_arg):
            cast_node = self._insert_format_cast_node(gm, weight_arg, node)

            new_args = list(node.args)
            new_args[1] = cast_node
            node.args = tuple(new_args)
            changed = True
            logger.info("Inserted format cast for single weight in grouped_matmul")

        if changed:
            self._mark_as_processed(node)

        return changed

    def _process_weight_quant_batchmatmul_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        weight_arg = node.args[1]

        if self._should_process_weight(weight_arg):
            cast_node = self._insert_format_cast_node(gm, weight_arg, node)

            new_args = list(node.args)
            new_args[1] = cast_node
            node.args = tuple(new_args)

            self._mark_as_processed(node)

            logger.info("Inserted format cast for weight in npu_weight_quant_batchmatmul")
            return True

        return False

    def _process_mla_prolog_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 5:
            return False

        weight_indices = [1, 2, 4]  # weight_dq, weight_uq_qr, weight_dkv_kr
        weight_names = ["weight_dq", "weight_uq_qr", "weight_dkv_kr"]

        changed = False
        new_args = list(node.args)

        for idx, weight_name in zip(weight_indices, weight_names):
            if idx < len(node.args):
                weight_arg = node.args[idx]

                if self._should_process_weight(weight_arg):
                    cast_node = self._insert_format_cast_node(gm, weight_arg, node)

                    new_args[idx] = cast_node
                    changed = True
                    logger.info(f"Inserted format cast for {weight_name} in npu_mla_prolog")

        if changed:
            node.args = tuple(new_args)
            self._mark_as_processed(node)

        return changed

    def _process_convert_weight_to_int4pack(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 1:
            return False

        weight_arg = node.args[0]

        if self._should_process_weight(weight_arg):
            cast_node = self._insert_format_cast_node(gm, weight_arg, node)

            new_args = list(node.args)
            new_args[0] = cast_node
            node.args = tuple(new_args)

            self._mark_as_processed(node)

            logger.info("Inserted format cast for weight in npu_convert_weight_to_int4pack")
            return True

        return False

    def _process_mm_all_reduce_base_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        x2_arg = node.args[1]

        if self._should_process_weight(x2_arg):
            cast_node = self._insert_format_cast_node(gm, x2_arg, node)

            new_args = list(node.args)
            new_args[1] = cast_node
            node.args = tuple(new_args)

            self._mark_as_processed(node)

            logger.info("Inserted format cast for x2 weight in npu_mm_all_reduce_base")
            return True

        return False

    def _process_grouped_matmul_finalize_routing_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        w_arg = node.args[1]

        if self._should_process_weight(w_arg):
            cast_node = self._insert_format_cast_node(gm, w_arg, node)

            new_args = list(node.args)
            new_args[1] = cast_node
            node.args = tuple(new_args)

            self._mark_as_processed(node)

            logger.info("Inserted format cast for w weight in torch_npu.npu_grouped_matmul_finalize_routing")
            return True

        return False

    def _process_mm_weights(self, gm: fx.GraphModule, node: fx.Node) -> bool:
        """Process torch.ops.aten.mm.default - both arguments support NZ format"""
        if self._is_already_processed(node):
            return False

        if len(node.args) < 2:
            return False

        changed = False
        new_args = list(node.args)

        if self._should_process_weight(node.args[0]):
            cast_node = self._insert_format_cast_node(gm, node.args[0], node)
            new_args[0] = cast_node
            changed = True
            logger.info("Inserted format cast for first argument in torch.ops.aten.mm.default")

        if self._should_process_weight(node.args[1]):
            cast_node = self._insert_format_cast_node(gm, node.args[1], node)
            new_args[1] = cast_node
            changed = True
            logger.info("Inserted format cast for second argument in torch.ops.aten.mm.default")

        if changed:
            node.args = tuple(new_args)
            self._mark_as_processed(node)

        return changed

    def process(self, gm: fx.GraphModule) -> bool:
        changed = False
        graph = gm.graph

        candidates = [node for node in graph.nodes if node.op == "call_function" and node.target in self.SUPPORTED_OPS_ND2NZ]

        operation_processors = {
            torch.ops.npu.npu_grouped_matmul.default: self._process_grouped_matmul_weights,
            torch.ops.npu.npu_quant_matmul.default: self._process_quant_matmul_weights,
            torch.ops.npu.npu_weight_quant_batchmatmul.default: self._process_weight_quant_batchmatmul_weights,
            torch.ops.npu.npu_mla_prolog.default: self._process_mla_prolog_weights,
            torch.ops.npu.npu_convert_weight_to_int4pack.default: self._process_convert_weight_to_int4pack,
            torch.ops.npu.npu_mm_all_reduce_base.default: self._process_mm_all_reduce_base_weights,
            torch.ops.npu.npu_grouped_matmul_finalize_routing: self._process_grouped_matmul_finalize_routing_weights,
            torch.ops.aten.mm.default: self._process_mm_weights,
        }

        for node in candidates:
            processor = operation_processors.get(node.target)
            if processor and processor(gm, node):
                changed = True

        if changed:
            gm.graph.lint()
            logger.info(
                f"FoldNdToNzFormat: Inserted format casts for {len([c for c in candidates if changed])} operations"
            )

        return changed
