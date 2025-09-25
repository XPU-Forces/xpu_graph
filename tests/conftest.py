import pytest
import torch


@pytest.fixture(autouse=True)
def reset_dynamo():
    try:
        torch._dynamo.reset()
        yield
    finally:
        torch._dynamo.reset()
