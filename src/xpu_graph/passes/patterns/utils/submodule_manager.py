import torch
import torch.fx

from xpu_graph.fx_utils import get_disable_fake_mode


def _gen_target_name(gm: torch.fx.GraphModule, target: str):
    *prefix, field = target.split(".")

    mod = gm
    for item in prefix:
        submod = getattr(mod, item, None)

        if submod is None:
            submod = torch.nn.Module()
            setattr(mod, item, submod)

        if not isinstance(submod, torch.nn.Module):
            raise ValueError(
                f"Cannot register submodule {target} on {mod}, because {getattr(mod, item)} is not a module"
            )

        mod = submod

    name_counter = 0
    indexed_field = f"{field}_{name_counter}"
    while hasattr(mod, indexed_field):
        name_counter += 1
        indexed_field = f"{field}_{name_counter}"
    return f"{target}_{name_counter}"


def register_new_submodule(
    gm: torch.fx.GraphModule,
    target: str,
    modcls,
    *,
    args: tuple = (),
    kwargs: dict = {},
):
    indexed_name = _gen_target_name(gm, target)

    disable_fake_mode = get_disable_fake_mode()
    with disable_fake_mode():
        gm.add_submodule(indexed_name, modcls(*args, **kwargs))

    return indexed_name
