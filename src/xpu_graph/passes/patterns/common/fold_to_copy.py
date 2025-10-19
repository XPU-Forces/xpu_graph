import torch
import torch.fx as fx
from torch.multiprocessing.reductions import StorageWeakRef

from xpu_graph.fx_utils import FxStage, has_storage
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.passes.patterns.utils.shape_utils import compatible_memory_format


class FoldToCopy(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]
    _pattern_group = PatternGroup.GROUP1

    def process(self, gm: fx.GraphModule):
        changed = False

        output_node: fx.Node = list(gm.graph.nodes)[-1]
        assert output_node.op == "output"
        output_storages = {
            StorageWeakRef(n.meta["val"].untyped_storage()) for n in output_node.all_input_nodes if has_storage(n)
        }

        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten._to_copy.default
            and has_storage(node)
            and StorageWeakRef(node.meta["val"].untyped_storage()) not in output_storages
        ]

        def _useless_to_copy(copy: fx.Node) -> bool:
            inp = copy.args[0]
            if "val" not in inp.meta or "val" not in copy.meta:
                return False
            if inp.meta["val"].dtype != copy.meta["val"].dtype:
                return False
            if "layout" in copy.kwargs:
                return False

            if inp.meta["val"].device != copy.meta["val"].device:
                return False
            if "pin_memory" in copy.kwargs or "non_blocking" in copy.kwargs:
                return False
            if "memory_format" in copy.kwargs:
                return copy.kwargs["memory_format"] == torch.preserve_format
            return True

        for _to_copy in candidates:
            if _useless_to_copy(_to_copy):
                changed = True
                _to_copy.replace_all_uses_with(_to_copy.args[0])
                gm.graph.erase_node(_to_copy)

        return changed
