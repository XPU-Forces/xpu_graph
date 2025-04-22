from xpu_graph.config import Target

def get_all_patterns(config) -> dict:
    if config.target == Target.mlu:
        from .mlu import get_all_patterns
        return get_all_patterns(config)
    elif config.target == Target.npu:
        from .npu import get_all_patterns
        return get_all_patterns(config)
    return {}
