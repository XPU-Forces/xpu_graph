import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mlu.manual_seed(seed)
    torch.mlu.manual_seed_all(seed)


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len


class MatMulModel(nn.Module):
    def __init__(self, in_features=64):
        super(MatMulModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, in_features))
        self.bias = nn.Parameter(torch.randn(in_features))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class MainModel(nn.Module):
    def __init__(self, in_features=64):
        super(MainModel, self).__init__()
        self.inner = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2 * in_features),
            nn.ReLU(),
            nn.Linear(in_features=2 * in_features, out_features=in_features),
        )

    def forward(self, x):
        return self.inner(x)


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


def dist_setup(rank, world_size, mesh_shape=None, mesh_dim_names=None):
    set_seed(12)
    if mesh_shape is None:
        mesh_shape = (world_size,)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    device_mesh = init_device_mesh("mlu", mesh_shape, mesh_dim_names=mesh_dim_names)
    torch.mlu.set_device(rank)
    return device_mesh


def cleanup():
    dist.destroy_process_group()


def get_dp_dataloader(rank, world_size):
    dataset = RandomDataset(size=64, length=1024)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
