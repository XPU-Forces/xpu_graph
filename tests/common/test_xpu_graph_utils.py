import os
from dataclasses import dataclass

import pytest

from xpu_graph.utils import get_bool_env_var, recursive_set_obj


@pytest.mark.parametrize(
    "val, expect",
    (
        ("0", False),
        ("1", True),
        ("true", True),
        ("false", False),
        ("on", True),
        ("off", False),
    ),
)
def test_get_bool_env_var(val, expect):
    env_name = "TEST_BOOL"
    os.environ[env_name] = val
    assert get_bool_env_var(env_name, not expect) == expect


class TestRecursiveSetObj:
    class Dummy:
        def __init__(self):
            self.a = None
            self.b = {"x": 1}

            @dataclass
            class Dummy:
                y: str = None

            self.c = Dummy()

    @pytest.mark.parametrize(
        "src_dict, assertion",
        (
            (
                {"a": 123, "b": {"x": 456}},
                lambda obj: obj.a == 123 and obj.b["x"] == 456,
            ),
            (
                {"c": {"y": 123}, "b": {"x": 456}},
                lambda obj: obj.c.y == 123 and obj.b["x"] == 456,
            ),
        ),
    )
    def test_recursive_set_obj(self, src_dict, assertion):
        tgt_obj = self.Dummy()
        recursive_set_obj(src_dict, tgt_obj)
        assert assertion(tgt_obj)
        ...
