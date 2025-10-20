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

        for clone in candidates:
            inp = clone.args[0]
            if "val" not in inp.meta or not isinstance(inp.meta["val"], torch.Tensor):
                continue
            # Note(chenyifan):
            #   We do not check memory_format compability for now, as input strides may get changed after other passes
            #   Thus, only preserve_format is allowed to fold; format-specified clone is always enforced
            target_memoryformat = clone.kwargs.get("memory_format", torch.preserve_format)
            if target_memoryformat == torch.preserve_format:
                changed = True
                clone.replace_all_uses_with(inp)
                gm.graph.erase_node(clone)

        return changed
