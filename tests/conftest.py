import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo():
    try:
        torch._dynamo.reset()
        yield
    finally:
        torch._dynamo.reset()


@pytest.fixture(scope="class", autouse=True)
def class_env_setup(request):
    if hasattr(request, "param"):
        env_vars = request.param

        with pytest.MonkeyPatch.context() as mp:
            for key, value in env_vars.items():
                mp.setenv(key, value)
            yield env_vars
    else:
        yield {}
