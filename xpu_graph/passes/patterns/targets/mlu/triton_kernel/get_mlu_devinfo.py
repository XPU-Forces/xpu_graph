import copy
import math
import os
from collections import namedtuple
from functools import lru_cache

import torch
import triton.backends.mlu.driver as driver
import triton.language as tl

DeviceProperties = namedtuple(
    "DeviceProperties",
    ["cluster_num", "cores_per_cluster", "total_cores", "max_cores", "max_nram_size"],
)


@lru_cache(maxsize=None)
def get_device_properties():
    _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
    cluster_num = _devprob.get("cluster_num")
    cores_per_cluster = _devprob.get("core_num_per_cluster")
    max_cores = cluster_num * cores_per_cluster
    total_cores = max_cores

    xpucore = os.getenv("XPUGRAPH_MLU_CORENUM", None)
    if xpucore != None:
        total_cores = int(xpucore)

    max_nram_size = int(_devprob.get("max_nram_size") * 0.8)

    return DeviceProperties(cluster_num, cores_per_cluster, total_cores, max_cores, max_nram_size)


# props = get_device_properties()
# print(f"Cluster Num: {props.cluster_num}, Cores per Cluster: {props.cores_per_cluster}, Total Cores: {props.total_cores}, Max nram size: {props.max_nram_size}")
