import torch
import torch.fx as fx

from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import check_cat_op

class FoldCat(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default
        ]

        for cat in candidates:
            inps = cat.args[0]
            if len(inps) == 1:
                changed = True

                inp = inps[0]
                cat.replace_all_uses_with(inp)
                gm.graph.erase_node(cat)

        gm.graph.lint()
        gm.recompile()
        return changed

class FoldCatCat(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        for node in reversed(gm.graph.nodes):
            if not check_cat_op(node):
                continue
            cat_axis = node.args[1]
            if node.meta == {}:
                continue
            if cat_axis == len(node.meta["tensor_meta"].shape) - 1:
                cat_axis = -1
            cat_input = []
            changed1 = False
            for m in node.args[0]:
                if check_cat_op(m):
                    cat_axis1 = m.args[1]
                    if cat_axis1 == len(m.meta["tensor_meta"].shape) - 1:
                        cat_axis1 = -1
                    if (len(m.users) == 1) and (cat_axis == cat_axis1):
                        cat_input += m.args[0]
                        changed1 = True
                    else:
                        cat_input.append(m)
                else:
                    cat_input.append(m)
            if changed1:
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