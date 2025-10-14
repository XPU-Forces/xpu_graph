import torch
import torch.fx as fx
from torch.multiprocessing.reductions import StorageWeakRef

from xpu_graph.fx_utils import FxStage, has_storage
from xpu_graph.passes.patterns.pattern import Pattern


class FoldClone(Pattern):
    _support_stages = [FxStage.inference]

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
            and node.target == torch.ops.aten.clone.default
            and has_storage(node)
            and StorageWeakRef(node.meta["val"].untyped_storage()) not in output_storages
        ]

        def _is_alias_of_output(node, output_storages):
            return StorageWeakRef(node.meta["val"].untyped_storage()) in output_storages

        for clone in candidates:
            inp = clone.args[0]
            if "tensor_meta" not in inp.meta:
                continue
            org_memoryformat = inp.meta["tensor_meta"].memory_format
            target_memoryformat = clone.kwargs["memory_format"] if "memory_format" in clone.kwargs else org_memoryformat
            if org_memoryformat == target_memoryformat:
                changed = True
                clone.replace_all_uses_with(inp)
                gm.graph.erase_node(clone)

        return changed
