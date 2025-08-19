import importlib


def get_all_patterns(config) -> dict:
    try:
        target_mod = importlib.import_module(f".{config.target.value}", __package__)
        return target_mod.get_all_patterns(config)
    except ImportError:
        return {}


def get_structure_replacements(config) -> dict:
    try:
        target_mod = importlib.import_module(f".{config.target.value}.structure_replacements", __package__)
        replacements = target_mod.get_structure_replacements(config)
    except ImportError:
        replacements = {}

    replacement_args = {}
    for pat_name, args in replacements.items():
        if isinstance(args, tuple):
            target_mod, constraint_fn = args
            replacement_args[pat_name] = {"target_mod": target_mod, "constraint_fn": constraint_fn}
        else:
            replacement_args[pat_name] = {"target_mod": args}
    return replacement_args
