import numpy as np
import pytest
import torch
from pytest import approx

import xpu_graph
from xpu_graph.config import XpuGraphConfig
from xpu_graph.test_utils import timeit

# TODO(liuyuan):make it test nightly for machine-monopolization.
# class TestCompileTime():
#     def setup_class(self, config=None):
#         config = (
#             XpuGraphConfig(is_training=False, debug=False)
#             if config is None or not isinstance(config, XpuGraphConfig)
#             else config
#         )
#         self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

#     @pytest.mark.parametrize(
#         "repeat_time, expected_time", [(128, 0.52)]
#     )
#     def test_compile_time(self, repeat_time, expected_time):
#         model = torch.nn.Sequential(
#             *[torch.nn.Linear(1024, 1024) for _ in range(repeat_time)]
#         )

#         @timeit
#         def compile_and_run():
#             compiled = torch.compile(model, backend=self.xpu_graph, dynamic=None)
#             compiled(torch.randn(1024, 1024))

#         elasped_times = []
#         for i in range(20):
#             _, elasped_time = compile_and_run()
#             elasped_times.append(elasped_time)
#         avg_time = np.mean(elasped_times)
#         print(avg_time)
#         assert avg_time == approx(expected_time, rel=0.2)


# if __name__ == '__main__':
#     config = XpuGraphConfig(is_training=False, debug=False)
#     test = TestCompileTime()
#     test.setup_class(config)
#     test.test_compile_time(10, 0.87)
