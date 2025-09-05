import torch

from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import need_xpu_graph_logs


def test_skip_pattern(caplog):
    with need_xpu_graph_logs():
        config = XpuGraphConfig(is_training=False, skip_patterns=["FoldAdd0", "FoldMul1", "Dummy"], debug=True)
        backend = XpuGraph(config)

    has_fold_add0 = False
    has_fold_sub0 = False
    has_fold_mul1 = False
    has_fold_div1 = False

    for group in backend._pass_manager.get_pattern_manager()._patterns.keys():
        for pattern in backend._pass_manager.get_pattern_manager()._patterns[group]:
            has_fold_add0 = has_fold_add0 or str(pattern) == "FoldAdd0"
            has_fold_sub0 = has_fold_sub0 or str(pattern) == "FoldSub0"
            has_fold_mul1 = has_fold_mul1 or str(pattern) == "FoldMul1"
            has_fold_div1 = has_fold_div1 or str(pattern) == "FoldDiv1"

    assert not has_fold_add0
    assert has_fold_sub0
    assert not has_fold_mul1
    assert has_fold_div1

    assert "xpu_graph skip builtin patterns: ['FoldAdd0', 'FoldMul1']" in caplog.text


if __name__ == "__main__":
    config = XpuGraphConfig(is_training=False, skip_patterns=["FoldAdd0", "FoldMul1", "Dummy"], debug=True)
    graph = XpuGraph(config)
