import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import check_cat_op


class FoldCat(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def _get_fold_result(self, gm: fx.GraphModule, src: fx.Node):
        return gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(src,),
        )

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [node for node in gm.graph.nodes if check_cat_op(node)[0]]

        for cat in candidates:
            inps = cat.args[0]
            if len(inps) == 1:
                changed = True

                with gm.graph.inserting_before(cat):
                    fold_res = self._get_fold_result(gm, inps[0])
                cat.replace_all_uses_with(fold_res)
                gm.graph.erase_node(cat)

        return changed


class FoldCatCat(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        for node in reversed(gm.graph.nodes):
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat:
                continue
            if node.meta == {}:
                continue
            if cat_axis == len(node.meta["val"].shape) - 1:
                cat_axis = -1
            cat_input = []
            foldable = False
            for inp in node.args[0]:
                is_input_cat, input_cat_axis = check_cat_op(inp)
                if is_input_cat:
                    if input_cat_axis == len(inp.meta["val"].shape) - 1:
                        input_cat_axis = -1
                    if len(inp.users) == 1 and cat_axis == input_cat_axis:
                        cat_input += inp.args[0]
                        foldable = True
                    else:
                        cat_input.append(inp)
                else:
                    cat_input.append(inp)
            if foldable:
                with gm.graph.inserting_before(node):
                    concat_node = gm.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(cat_input, cat_axis),
                        name=node.name + "_1",
                    )
                node.replace_all_uses_with(concat_node)
                gm.graph.erase_node(node)
                changed = True

        return changed
