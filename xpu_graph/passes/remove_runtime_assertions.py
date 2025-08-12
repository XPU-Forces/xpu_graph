import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage

from .optimizer import Optimizer

aten = torch.ops.aten

"""
After torch 2.6, when AOTConfig.export is True, FunctionalTensor will add runtime assertions when processing Tensor.to operations
This would mutates the node usage info and prevents passes and inductor inner_compile from working correctly
As a workaround, we remove assertions before other passes
"""


class RemoveAssertions(Optimizer):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        if not hasattr(aten, "_assert_tensor_metadata"):
            return False
        changed = False
        for node in gm.graph.find_nodes(op="call_function", target=torch.ops.aten._assert_tensor_metadata.default):
            gm.graph.erase_node(node)
            changed = True
        return changed
