import operator

import torch
import torch.nn.functional as F
from torch import fx, nn

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.passes.patterns.utils.shape_utils import SymShapeManager

from ...utils.combo_utils import COMBINE_MM_WIDTH, find_dep, partially_topo_sort
from .combo_matmul_utils import *


def all_same_tensor(tensor_list):
    if not tensor_list:
        return True
    first = tensor_list[0]
    return all(t is first for t in tensor_list)


class FusedCombineBmm(nn.Module):
    def forward(self, input_list, weight_list, bias_list, act: str, trans_a: bool, trans_b: bool, use_groupgemm: bool):
        if use_groupgemm:
            output = self.forward_groupgemm(input_list, weight_list, bias_list, trans_a, trans_b)
        else:
            output = self.forward_bmm(input_list, weight_list, bias_list, trans_a, trans_b)

        # Apply activation function if specified

        activations = {
            "relu": torch.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "sigmoid": torch.sigmoid,
        }
        if act in activations:
            act_func = activations.get(act.lower())
            output = act_func(output)

        return output

    def forward_bmm(self, input_list, weight_list, bias_list, trans_a, trans_b):
        input_batch = torch.stack(input_list, dim=0)
        if trans_a:
            input_batch = input_batch.transpose(-1, -2)

        weight_batch = torch.stack(weight_list, dim=0)
        if trans_b:
            weight_batch = weight_batch.transpose(-1, -2)

        input_shape = input_batch.shape
        weight_shape = weight_batch.shape
        M = input_shape[-2]
        K = weight_shape[-2]
        N = weight_shape[-1]

        if len(weight_shape) == 4:
            input_batch = input_batch.view(-1, M, K)
            weight_batch = weight_batch.view(-1, K, N)

        if bias_list[0] is not None:
            bias_batch = torch.stack(bias_list)
            # bias: [N]
            if len(bias_list[1].shape) == 1:
                # bias_batch: [T, 1, N]
                bias_batch = bias_batch.unsqueeze(1)
            if len(bias_batch.shape) == 4:
                bias_batch = bias_batch.view(-1, M, N)
            output = torch.bmm(input_batch, weight_batch) + bias_batch
        else:
            output = torch.bmm(input_batch, weight_batch)

        if len(weight_shape) == 4:
            output = output.view(weight_shape[0], input_shape[1], M, N)
        return output

    def forward_groupgemm(self, input_list, weight_list, bias_list, trans_a: bool, trans_b: bool):
        processed_inputs = [i.contiguous() if not i.is_contiguous() else i for i in input_list]
        processed_weights = [w.contiguous() if not w.is_contiguous() else w for w in weight_list]
        args = [processed_inputs, processed_weights]
        kwargs = {"trans_a": trans_a, "trans_b": trans_b}
        if bias_list[0] is not None:
            if len(bias_list[0].shape) == 1:
                kwargs["bias"] = bias_list
            else:
                beta = [1.0] * len(input_list)
                kwargs["beta"] = beta
                kwargs["c"] = bias_list
        output = torch.ops.torch_mlu.grouped_gemm(*args, **kwargs)
        if isinstance(output, list):
            output = torch.stack(output, dim=0)
        return output


class FusedCombineMatMul(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1
    _support_stages = [
        FxStage.inference,
        # NOTE: though combo_mm seems applicable to training mode, but it is not fully verified yet, disable it for now
        # FxStage.pregrad,
        # FxStage.forward,
        # FxStage.backward,
    ]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        graph_module.add_submodule("fused_combo_bmm", FusedCombineBmm())
        sym_shape_manager = SymShapeManager(graph_module.graph)
        # split mm by difference module
        target_module = [
            torch.ops.aten.mm.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.linear.default,
            "custom_batch_dense_layer_replacement",
            "custom_dense_layer_replacement",
        ]
        for module in target_module:
            candidates = [
                node
                for node in graph_module.graph.nodes
                if (node.op == "call_function" or node.op == "call_module") and node.target == module
            ]
            if len(candidates) < COMBINE_MM_WIDTH:
                continue
            changed = changed | self.combo_matmul(graph_module, candidates, sym_shape_manager, COMBINE_MM_WIDTH)

        return changed

    def replace_node(self, graph_module, mm_descs):
        new_input = [desc.input1 for desc in mm_descs]
        new_weight = [desc.input2 for desc in mm_descs]
        new_bias = [desc.bias for desc in mm_descs]
        trans_a = mm_descs[0].input1_trans
        trans_b = mm_descs[0].input2_trans
        act = mm_descs[0].act

        last_node = max(desc.node for desc in mm_descs if isinstance(desc.node, fx.Node))
        with graph_module.graph.inserting_before(last_node):
            if (
                self._current_stage == FxStage.inference
                and hasattr(torch.ops.torch_mlu, "grouped_gemm")
                and len(new_input[0].meta["val"].shape) == 2
            ):
                use_groupgemm = True
            else:
                use_groupgemm = False
            new_node = graph_module.graph.call_module(
                "fused_combo_bmm",
                args=(new_input, new_weight, new_bias, act, trans_a, trans_b, use_groupgemm),
            )
            unbind_node = graph_module.graph.call_function(torch.ops.aten.unbind.int, args=(new_node,))
            new_nodes = []
            for idx in range(len(mm_descs)):
                new_n = graph_module.graph.call_function(operator.getitem, args=(unbind_node, idx))
                new_nodes.append(new_n)

            for desc, new_n in zip(mm_descs, new_nodes):
                if desc.extra_match == True:
                    next_node = next(iter(desc.node.users))
                    next_node.replace_all_uses_with(new_n)
                else:
                    desc.node.replace_all_uses_with(new_n)
                partially_topo_sort(new_n, insert_after=new_nodes[-1])

    def combo_matmul(self, graph_module, candidates, sym_shape_manager, combine_mm_width):
        changed = False
        group_by_shape = {}
        # split mm by input&weight&bias's shape and activation mode
        for n in candidates:
            mm_desc = get_node_desc(n)
            if mm_desc == None:
                continue

            key = (
                mm_desc.input1_trans,
                mm_desc.input2_trans,
                tuple(sym_shape_manager.rebind_shape(mm_desc.input1_shape)),
                tuple(sym_shape_manager.rebind_shape(mm_desc.input2_shape)),
                ((None if mm_desc.bias == None else tuple(sym_shape_manager.rebind_shape(mm_desc.bias_shape))),),
                mm_desc.act,
            )
            if key not in group_by_shape:
                group_by_shape[key] = []
            group_by_shape[key].append(mm_desc)

        for group_nodes in group_by_shape.values():
            group_by_input = find_dep(group_nodes, has_mm_dependency)
            for mm_descs in group_by_input:
                if len(mm_descs) < combine_mm_width:
                    continue
                changed = True
                self.replace_node(graph_module, mm_descs)
        return changed
