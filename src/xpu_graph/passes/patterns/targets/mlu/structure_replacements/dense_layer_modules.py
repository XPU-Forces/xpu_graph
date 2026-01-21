import torch


class DenseLayerModule(torch.nn.Module):
    def __init__(self, fast_act):
        super().__init__()
        self.fast_act = fast_act

    def forward(self, inputs, weight, weight_trans, bias, act):
        import torch_mlu_ops

        act = act

        # input last dim must be contiguous.
        if inputs.stride()[-1] != 1:
            inputs = inputs.contiguous()

        m = inputs.shape[0]
        n = weight.shape[-1] if not weight_trans else weight.shape[-2]
        if bias is not None:
            bias_shape = bias.shape
            if len(bias_shape) == 2 and bias_shape[0] == 1:
                bias = bias.view(-1)
                bias_shape = bias.shape
            if len(bias_shape) == 1:
                if bias.shape[0] != weight.shape[0]:
                    bias = bias.broadcast_to((n,)).contiguous()
                output = torch_mlu_ops.matmul(
                    inputs,
                    weight,
                    bias,
                    None,
                    act,
                    1.0,
                    0.0,
                    self.fast_act,
                    False,
                    trans_b=weight_trans,
                )
                return output

        # bias 2d or None
        if bias is not None and bias.shape != torch.Size([m, n]):
            bias = bias.broadcast_to((m, n)).contiguous()
        output = torch_mlu_ops.matmul(
            inputs,
            weight,
            None,
            bias,
            act,
            1.0,
            0.0 if bias is None else 1.0,
            self.fast_act,
            False,
            trans_b=weight_trans,
        )
        return output


def can_fuse_custom_denselayer(inputs, weight, weight_trans, bias, act_str):
    if act_str not in ["gelu", "relu", "silu", "sigmoid", "none"]:
        return False

    if act_str == "sigmoid":
        # TODO(jyj): waiting for tmo version update
        return False

    return True


class BatchDenseLayerModule(torch.nn.Module):
    def forward(self, inputs, weight, weight_trans, bias, act):
        import torch_mlu_ops

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if bias is not None:
            if not bias.is_contiguous():
                bias = bias.contiguous()
        b, m, _ = inputs.shape
        n = weight.shape[-1] if not weight_trans else weight.shape[-2]
        if bias is not None and bias.shape == torch.Size([b, m, n]):
            residual = bias
            bias = None
            beta = 1.0
        else:
            residual = None
            beta = 0.0

        if bias is not None and bias.shape != torch.Size([b, 1, n]):
            bias = bias.broadcast_to((b, 1, n)).contiguous()

        dtype = inputs.dtype

        output = torch_mlu_ops.batch_matmul(
            inputs,
            weight,
            residual,
            1.0,
            beta,
            1.0,
            1.0,
            False,
            weight_trans,
            None,
            bias,
            act,
            dtype,
        )

        return output


def can_fuse_custom_batch_denselayer(inputs, weight, weight_trans, bias, act_str):
    if act_str not in ["gelu", "silu", "none"]:
        return False

    return True
