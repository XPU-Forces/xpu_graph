import torch
import torch.nn.functional as F


class DefaultRMSNorm(torch.nn.Module):
    def forward(self, x, weight, eps, dtype):
        input_dtype = x.dtype
        x = x.to(dtype)
        if weight is not None:
            weight = weight.to(dtype)
        return F.rms_norm(x, x.shape[-1:], weight=weight, eps=eps).to(input_dtype)
