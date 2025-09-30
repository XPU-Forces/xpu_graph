import torch
import torch.nn.functional as F
import torch_npu


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weight, epsilon):
        if weight is not None and weight.dtype != inputs.dtype:
            weight = weight.to(inputs.dtype)
        return torch_npu.npu_rms_norm(inputs, weight, epsilon)[0]
