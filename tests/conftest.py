import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo():
    try:
        torch._dynamo.reset()
        yield
    finally:
        torch._dynamo.reset()


@pytest.fixture(params=["0", "1"], autouse=True)
def env_dispatch(request, monkeypatch):
    monkeypatch.setenv("XPUGRAPH_FALLBACK_LEGACY_DISPATCH", request.param)
    yield request.param
