import pytest
import torch


@pytest.fixture(autouse=True)
def reset_npu_mempool():
    try:
        torch.npu.empty_cache()
        yield
    finally:
        pass
        torch.npu.empty_cache()
