import torch
import torch.nn.functional as F
import torch_npu


class LayerNormModule(torch.nn.Module):
    def forward(self, inputs, weight, bias, epsilon):
        if weight is not None and weight.dtype != inputs.dtype:
            weight = weight.to(inputs.dtype)
        if bias is not None and bias.dtype != inputs.dtype:
            bias = bias.to(inputs.dtype)
        # NPU直接支持F.layer_norm
        return F.layer_norm(inputs, inputs.shape[-1:], weight, bias, epsilon)