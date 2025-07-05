import torch
import torch.nn.functional as F


class DefaultRMSNorm(torch.nn.Module):
    def forward(self, x, weight, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        if weight is not None:
            weight = weight.to(torch.float32)
        return F.rms_norm(x, x.shape[-1:], weight=weight, eps=eps).to(input_dtype)
