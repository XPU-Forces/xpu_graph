import torch
import torch.fx as fx

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import is_one_like, is_zero_like
from xpu_graph.passes.patterns.utils.get_binary_fold_result import (
    get_binary_fold_result,
)


def is_constant_tensor_and_shape_reducible(where: fx.Node, inp, other):
    if not isinstance(other, fx.Node) or not isinstance(other.meta["val"], torch.Tensor):
        return None
    if where.meta["val"].shape != other.meta["val"].shape:
        return None
    if not isinstance(inp, fx.Node) or inp.op != "call_function":
        return None
    if inp.meta["val"].shape == torch.Size([1]):
        return None
    aten = torch.ops.aten
    if inp.target in (aten.ones.default, aten.ones_like.default):
        return 1
    elif inp.target in (aten.zeros.default, aten.zeros_like.default):
        return 0
    elif inp.target in (aten.full.default, aten.full_like.default):
        return inp.args[1]
    else:
        return None


class FoldWhere(Pattern):
    _support_stages = [
        FxStage.inference,
        FxStage.pregrad,
        FxStage.forward,
        FxStage.backward,
    ]

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node for node in gm.graph.nodes if node.op == "call_function" and node.target == torch.ops.aten.where.self
        ]

        for where in candidates:
            inp = where.args[1]
            other = where.args[2]
            if (
                (inp == other)
                or (is_one_like(inp) and is_one_like(other))
                or (is_zero_like(inp) and is_zero_like(other))
            ):
                with gm.graph.inserting_before(where):
                    res = get_binary_fold_result(gm, inp, where.meta)

                if res is not None:
                    where.replace_all_uses_with(res)
                    gm.graph.erase_node(where)
                    changed = True
            else:
                inp_scalar = is_constant_tensor_and_shape_reducible(where, inp, other)
                other_scalar = is_constant_tensor_and_shape_reducible(where, other, inp)
                if inp_scalar is not None:
                    with gm.graph.inserting_before(where):
                        res = gm.graph.call_function(
                            torch.ops.aten.full.default,
                            args=([1], inp_scalar),
                            kwargs={"device": inp.meta["val"].device, "dtype": inp.meta["val"].dtype},
                        )
                    where.update_arg(1, res)
                    changed = True
                elif other_scalar is not None:
                    with gm.graph.inserting_before(where):
                        res = gm.graph.call_function(
                            torch.ops.aten.full.default,
                            args=([1], other_scalar),
                            kwargs={"device": other.meta["val"].device, "dtype": other.meta["val"].dtype},
                        )
                    where.update_arg(2, res)
                    changed = True

        return changed
