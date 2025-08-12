import logging
import time

import torch
from torch.testing._internal.common_utils import TestCase

from .cache import XpuGraphCache
from .compiler import XpuGraph
from .utils import logger


def is_similar(result, expected, rtol=0.01, atol=0.01):
    return result.shape == expected.shape and torch.allclose(result, expected, rtol, atol)


def maybe_similar(result, expected, rtol=0.01, atol=0.01):
    if result is None or expected is None:
        return result is None and expected is None
    return is_similar(result, expected, rtol, atol)


def aggregate_similar(result, expected, rtol=0.01, atol=0.01):
    if isinstance(result, tuple) or isinstance(expected, tuple):
        if not isinstance(result, tuple) or not isinstance(expected, tuple):
            return False
        if len(result) != len(expected):
            return False
        return all([aggregate_similar(r, e, rtol, atol) for r, e in zip(result, expected)])
    else:
        return is_similar(result, expected, rtol, atol)


def assertTensorsEqual(
    a,
    b,
    prec=None,
    message="",
    allow_inf=False,
    use_MSE=False,
    use_RAE=False,
    use_RMA=False,
):
    tc = TestCase()
    if a.dtype == torch.bool:
        a = a.float()
    if b.dtype == torch.bool:
        b = b.float()
    epsilon = 1.0 / 16384
    tc.assertEqual(a.size(), b.size(), message)
    assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), "a and b are need be torch tensor."
    if a.numel() > 0:
        # check that NaNs are in the same locations
        nan_mask = a != a
        tc.assertTrue(torch.equal(nan_mask, b != b), message)
        diff = a - b
        diff[nan_mask] = 0
        a = a.clone()
        b = b.clone()
        a[nan_mask] = 0
        b[nan_mask] = 0
        # inf check if allow_inf=True
        if allow_inf:
            inf_mask = (a == float("inf")) | (a == float("-inf"))
            tc.assertTrue(
                torch.equal(inf_mask, (b == float("inf")) | (b == float("-inf"))),
                message,
            )
            diff[inf_mask] = 0
            a[inf_mask] = 0
            b[inf_mask] = 0
        # TODO: implement abs on CharTensor
        if diff.is_signed() and "CharTensor" not in diff.type():
            diff = diff.abs()
        if use_MSE:
            diff = diff.abs().pow(2).sum()
            a_pow_sum = a.pow(2).sum()
            if diff <= (2 * epsilon) * (2 * epsilon):
                diff = 0.0
            if a_pow_sum <= epsilon:
                a_pow_sum = a_pow_sum + epsilon
            diff = torch.div(diff, (a_pow_sum * 1.0))
            tc.assertLessEqual(diff.sqrt(), prec, message)
        elif use_RAE:
            diff = diff.abs().sum()
            a_sum = a.abs().sum()
            if a_sum == 0:
                tc.assertEqual(a, b, message)
            else:
                diff = torch.div(diff, a_sum)
                tc.assertLessEqual(diff, prec, message)
        elif use_RMA:
            a_mean = a.abs().mean()
            b_mean = b.abs().mean()
            if a_mean == 0:
                tc.assertEqual(a, b, message)
            else:
                diff = torch.div((a_mean - b_mean).abs(), a_mean)
                tc.assertLessEqual(diff, prec, message)
        else:
            max_err = diff.max()
            tc.assertLessEqual(max_err, prec, message)


class need_xpu_graph_logs:
    def __init__(self):
        xpugraph_logger = logging.getLogger("xpu_graph")
        self.original_propagate = xpugraph_logger.propagate
        self.original_level = xpugraph_logger.level

    def __enter__(self):
        xpugraph_logger = logging.getLogger("xpu_graph")
        xpugraph_logger.propagate = True
        xpugraph_logger.setLevel(logging.DEBUG)

    def __exit__(self, exc_type, exc_val, exc_tb):
        xpugraph_logger = logging.getLogger("xpu_graph")
        xpugraph_logger.propagate = self.original_propagate
        xpugraph_logger.setLevel(self.original_level)


class skip_xpu_graph_cache:
    def __init__(self, xpu_graph_backend: XpuGraph):
        self.backend = xpu_graph_backend
        self.cache = xpu_graph_backend._cache

    def __enter__(self):
        # Use base cache to skip save/load
        torch._dynamo.reset_code_caches()
        self.backend._cache = XpuGraphCache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend._cache = self.cache


def timeit(func, *args, **kwargs):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start

    return wrap
