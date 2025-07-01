import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_mlu
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import xpu_graph
from xpu_graph.config import OptLevel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    torch.mlu.manual_seed_all(seed)


set_seed(12)


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


class MatMulModel(nn.Module):
    def __init__(self, in_features=10):
        super(MatMulModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, in_features))
        self.bias = nn.Parameter(torch.randn(in_features))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


def set_dist_env():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        import socket
        from contextlib import closing

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("localhost", 0))
            port = s.getsockname()[1]

        os.environ["MASTER_PORT"] = str(port)

    print(f'Setting master: {os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}')


def train_setup(rank, world_size):
    dist.init_process_group("cncl", rank=rank, world_size=world_size)
    torch.mlu.set_device(rank)


def cleanup():
    dist.destroy_process_group()
