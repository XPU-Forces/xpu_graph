import torch

from tests.common.test_models import SimpleModel, compare_inference
from xpu_graph import OptLevel, XpuGraph, XpuGraphConfig, XpuGraphLocalCache
from xpu_graph.cache import SerializableGraphModule
from xpu_graph.test_utils import need_xpu_graph_logs

device = "cpu"
dtype = torch.float32


def test_xpugraph_cache(caplog, tmp_path):
    # Note: currently, we do not guard on **freezed params**
    #       thus serialize a freeze artifact but with different parameters would result in wrong result
    infer_config = XpuGraphConfig(is_training=False, opt_level=OptLevel.level2, freeze=False)
    infer_backend = XpuGraph(infer_config, XpuGraphLocalCache(tmp_path))

    with need_xpu_graph_logs():
        for bsz in [8, 10, 20]:
            compare_inference(device, dtype, SimpleModel, infer_backend, bsz=bsz, dynamic=None)

        torch._dynamo.reset()

        for bsz in [8, 10, 20]:
            compare_inference(device, dtype, SimpleModel, infer_backend, bsz=bsz, dynamic=None)

    assert caplog.text.count("Save cache in location") == 2, "Should save cache twice"
    assert caplog.text.count("Use cache in location") == 2, "Should use cache twice"


if __name__ == "__main__":
    test_xpugraph_cache(None, "/tmp/test_xpu_graph_cache")
