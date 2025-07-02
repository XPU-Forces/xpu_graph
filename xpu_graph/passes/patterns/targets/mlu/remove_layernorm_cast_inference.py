import torch
import torch.fx as fx

from xpu_graph.config import OptLevel
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import AutoMatchPattern
from xpu_graph.passes.patterns.utils.check_ops import check_op


class RemoveLayerNormCastForward(AutoMatchPattern):
    _opt_level = OptLevel.level2
    _support_stages = [FxStage.inference]

    def rewriter(self, gm: fx.GraphModule, rule_name: str, node_map: dict) -> bool:
        assert len(node_map) == 4

        pre_cast = node_map["pre_cast"]
        layernorm = node_map["layernorm"]
        getitem = node_map["getitem"]
        post_cast = node_map["post_cast"]

        def _can_remove(pre_cast: fx.Node, post_cast: fx.Node) -> bool:
            inp = pre_cast.args[0]
            if inp.meta["val"].dtype not in (
                torch.bfloat16,
                torch.float16,
            ):
                return False
            if pre_cast.kwargs["dtype"] != torch.float:
                return False
            if post_cast.kwargs["dtype"] != inp.meta["val"].dtype:
                return False
            return True

        if not _can_remove(pre_cast, post_cast):
            return False

        layernorm.replace_input_with(layernorm.args[0], pre_cast.args[0])
        if check_op(layernorm.args[2], torch.ops.aten._to_copy.default) and _can_remove(layernorm.args[2], post_cast):
            layernorm.replace_input_with(layernorm.args[2], layernorm.args[2].args[0])
        if check_op(layernorm.args[3], torch.ops.aten._to_copy.default) and _can_remove(layernorm.args[3], post_cast):
            layernorm.replace_input_with(layernorm.args[3], layernorm.args[3].args[0])
        post_cast.replace_all_uses_with(getitem)

        return True
