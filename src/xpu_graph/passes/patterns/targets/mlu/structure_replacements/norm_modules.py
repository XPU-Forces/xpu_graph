import torch


class RMSNormModule(torch.nn.Module):
    def forward(self, inputs, weights, epsilon):
        import torch_mlu_ops

        if weights is not None and weights.dtype != inputs.dtype:
            weights = weights.to(inputs.dtype)
        return torch_mlu_ops.fused_rms_norm(inputs, None, weights, None, None, epsilon, False)


class LayerNormModule(torch.nn.Module):
    def forward(self, inputs, weights, bias, epsilon):
        import torch.nn.functional as F

        if weights is None and weights.dtype != inputs.dtype:
            weights = weights.to(inputs.dtype)
        if bias is None and bias.dtype != inputs.dtype:
            bias = bias.to(inputs.dtype)
        return F.layer_norm(inputs, inputs.shape[-1:], weights, bias, epsilon)
