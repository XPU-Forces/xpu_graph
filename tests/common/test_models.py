import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class SliceCatModel(nn.Module):
    def __init__(self, input_dim):
        super(SliceCatModel, self).__init__()
        self.fc = nn.Linear(input_dim, 16)

    def forward(self, x):
        x = self.fc(x)
        return torch.cat([-x[..., 8:], x[..., :8]], 1).sum(dim=-1)


class InplaceModel(nn.Module):
    def __init__(self, input_dim):
        super(InplaceModel, self).__init__()
        self.fc = nn.Linear(input_dim, 16)

    def forward(self, x):
        x = self.fc(x)
        y = x.clone()
        x.add_(1)
        z = x + y
        return z.sum(dim=-1)


class ConstantInplaceModel(nn.Module):
    def __init__(self, input_dim):
        super(ConstantInplaceModel, self).__init__()
        self.fc = nn.Linear(input_dim, 16)

    def forward(self, x):
        x = self.fc(x)
        indices = x.sum(dim=-1).nonzero().squeeze(-1)
        y = x[indices].sum(-1)
        max_len = indices.max() + 1
        zeros = torch.zeros(max_len, dtype=y.dtype)
        zeros = zeros.scatter_(0, indices, y)
        result = torch.cat([zeros, torch.zeros(x.shape[0] - max_len, dtype=zeros.dtype)], dim=0)
        return result


all_models = [SimpleModel, SliceCatModel, InplaceModel, ConstantInplaceModel]
