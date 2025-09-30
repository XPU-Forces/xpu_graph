import torch
import torch.fx as fx
from torch import SymInt

from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.shape_utils import SymShapeManager


class ChangeTensorLike(Pattern):
    _support_stages = [FxStage.inference, FxStage.pregrad]

    def process(self, gm: fx.GraphModule):
        changed = False
        sym_shape_manager = SymShapeManager(gm.graph)

        tensor_like_map = {
            torch.ops.aten.ones_like.default: torch.ops.aten.ones.default,
            torch.ops.aten.zeros_like.default: torch.ops.aten.zeros.default,
            torch.ops.aten.full_like.default: torch.ops.aten.full.default,
        }

        candidates = [node for node in gm.graph.nodes if node.op == "call_function" and node.target in tensor_like_map]

        for like_node in candidates:
            template_node = like_node.args[0]

            if "val" not in template_node.meta:
                continue

            template_shape = sym_shape_manager.rebind_shape(template_node.meta["val"].shape)
            # do not change if target shape cannot be rebinded
            if template_shape is None:
                continue

            fill_value_arg = None
            if like_node.target == torch.ops.aten.full_like.default:
                fill_value_arg = like_node.args[1]
                # Note: symbolic fill_value does not matter actually
                # if not isinstance(fill_value_arg, (int, float, bool)):
                #     continue

            changed = True
            with gm.graph.inserting_before(like_node):
                new_args = [template_shape]
                if fill_value_arg is not None:
                    new_args.append(fill_value_arg)

                # according to https://docs.pytorch.org/docs/stable/generated/torch.full_like.html#torch-full-like
                # torch.full_like(input, fill_value) is equivalent to
                # torch.full(input.size(), fill_value, dtype=input.dtype, layout=input.layout, device=input.device)
                # currently we do not handle "memory_format"
                new_kwargs = {
                    "dtype": like_node.kwargs.get("dtype", template_node.meta["val"].dtype),
                    "device": like_node.kwargs.get("device", template_node.meta["val"].device),
                    "layout": like_node.kwargs.get("layout", template_node.meta["val"].layout),
                }
                if "pin_memory" in like_node.kwargs:
                    new_kwargs["pin_memory"] = like_node.kwargs["pin_memory"]

                new_tensor_node = gm.graph.call_function(
                    tensor_like_map[like_node.target],
                    args=tuple(new_args),
                    kwargs=new_kwargs,
                )

            like_node.replace_all_uses_with(new_tensor_node)
            gm.graph.erase_node(like_node)

        return changed
