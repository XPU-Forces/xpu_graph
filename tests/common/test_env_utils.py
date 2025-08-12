import os

import pytest

from xpu_graph.utils import get_bool_env_var


class TestEnvUtils:
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
    def test_get_bool_env_var(self, val, expect):
        env_name = "TEST_BOOL"
        os.environ[env_name] = val
        assert get_bool_env_var(env_name, not expect) == expect
