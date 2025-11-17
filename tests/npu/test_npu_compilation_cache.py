import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from xpu_graph import Target
from xpu_graph.backends.npu import NpuSerializableArtifact
from xpu_graph.cache import XpuGraphLocalCache
from xpu_graph.compiler import XpuGraphCompiler
from xpu_graph.fx_utils import FxStage
from xpu_graph.test_utils import need_xpu_graph_logs


@pytest.fixture
def clear_cache_dir():
    import shutil
    from glob import glob

    from xpu_graph.config import get_cache_dir

    # NOTE(liuyuan): clear all cache
    for dir in glob("/tmp/xpu_graph_*"):
        shutil.rmtree(dir)


class TestNpuCompilationCache:
    def setup_method(self):
        self.compiler = XpuGraphCompiler()
        self.inputs = (torch.empty(1024).npu(), torch.empty(1024).npu())
        self.stage = FxStage.inference

    @pytest.mark.parametrize(
        "compiler_setting",
        (
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge", "mode": "reduce-overhead"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(False)
            .done(),
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(False)
            .done(),
        ),
        ids=("aclgraph", "ge"),
    )
    def test_cache(self, compiler_setting, clear_cache_dir):
        cache = XpuGraphLocalCache("/tmp/xpu_graph_whhh")
        compiler_setting(self)
        ori_gm = []
        compiled_gm = []

        def dummy(x, y):
            return x + y

        def hijack_backend(gm, *args, **kwargs):
            ori_gm.append(gm)
            compiled_gm.append(self.compiler._compiler(gm, *args, **kwargs))
            return compiled_gm[0]

        compiled_module = torch.compile(dummy, backend=hijack_backend, dynamic=False)
        compiled_module(*self.inputs)

        cache_key = cache.cache_key(
            ori_gm[0],
            self.inputs,
            self.compiler._config,
            self.stage,
        )

        cache.save_artifact(cache_key, NpuSerializableArtifact(compiled_gm[0]._compiled_func))
        compiled_module = cache.load_artifact(cache_key).artifact
        assert torch.allclose(compiled_module(*self.inputs)[0], dummy(*self.inputs), equal_nan=True)

    @pytest.mark.parametrize(
        "compiler_setting",
        (
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge", "mode": "reduce-overhead"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(True)
            .done(),
            lambda self: self.compiler.set_target(Target.npu)
            .set_vendor_compiler_config({"compiler": "ge"})
            .set_debug(True)
            .set_is_training(False)
            .set_enable_cache(True)
            .done(),
        ),
        ids=("aclgraph", "ge"),
    )
    def test_compile_pipeline(self, compiler_setting, clear_cache_dir, caplog):
        compiler_setting(self)

        class Model(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight = torch.nn.Parameter(torch.empty(1024))

            def forward(self, x, y):
                return x + y + self.weight

        model = Model().eval().npu()

        with need_xpu_graph_logs():
            compiled = self.compiler.compile(model, dynamic=False)
            compiled(*self.inputs)

            compiler_setting(self)
            compiled = self.compiler.compile(model, dynamic=False)
            compiled(*self.inputs)

            assert "Use cache in location" in caplog.text
