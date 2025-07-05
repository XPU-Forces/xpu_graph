import torch
import torch.nn.functional as F
import torch_npu


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weight, epsilon, dtype):
        return torch_npu.npu_rms_norm(inputs, weight, epsilon)[0]
