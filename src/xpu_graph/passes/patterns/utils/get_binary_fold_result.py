from typing import Any, Dict, Optional, Union

import torch
import torch.fx as fx

from .shape_utils import same_shape


def get_binary_fold_result(
    gm: fx.GraphModule, inp: Union[int, float, fx.Node], target_meta: Dict[str, Any]
) -> Optional[fx.Node]:
    scalar_tup = (
        int,
        float,
    )
    assert type(inp) in scalar_tup or isinstance(inp, fx.Node), "get_binary_fold_result input error"

    if type(inp) in scalar_tup:
        if any(isinstance(s, torch.SymInt) for s in target_meta["val"].shape):
            # FIXME: use shape env to get the real shape
            return None
        return gm.graph.call_function(
            torch.ops.aten.full.default,
            args=(target_meta["val"].shape, inp),
            kwargs={
                "dtype": target_meta["val"].dtype,
                "device": target_meta["val"].device,
            },
        )
    else:
        if inp.meta["val"].dtype == target_meta["val"].dtype:
            cast = inp
        else:
            cast = gm.graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(inp,),
                kwargs={
                    "dtype": target_meta["val"].dtype,
                },
            )
        if same_shape(inp.meta["val"].shape, target_meta["val"].shape):
            expand = cast
        else:
            if any(isinstance(s, torch.SymInt) for s in target_meta["val"].shape):
                # FIXME: use shape env to get the real shape
                return None
            expand = gm.graph.call_function(
                torch.ops.aten.expand.default,
                args=(
                    cast,
                    target_meta["val"].shape,
                ),
            )
        copy = gm.graph.call_function(
            torch.ops.aten.clone.default,
            args=(expand,),
            kwargs={
                "memory_format": torch.contiguous_format,
            },
        )
        return copy
