import torch


class DenseLayerModule(torch.nn.Module):
    def __init__(self, fast_act):
        super().__init__()
        self.fast_act = fast_act

    def forward(self, inputs, weight, weight_trans, bias, act):
        import torch_mlu_ops

        # TODO(jyj): waiting for tmo version update
        tmp_act = act
        if act == "sigmoid":
            tmp_act = "none"

        # input last dim must be contiguous.
        if inputs.stride()[-1] != 1:
            inputs = inputs.contiguous()

        if bias != None:
            if isinstance(bias, int):
                dim = weight.shape[1] if weight_trans == False else weight.shape[0]
                bias = torch.tensor([bias] * dim, device=inputs.device, dtype=inputs.dtype)
            bias_shape = bias.shape
            if (len(bias_shape) == 2) & (bias_shape[0] == 1):
                bias = bias.view(-1)
                bias_shape = bias.shape
            if len(bias_shape) == 1:
                output = torch_mlu_ops.matmul(
                    inputs,
                    weight,
                    bias,
                    None,
                    tmp_act,
                    1.0,
                    0.0,
                    self.fast_act,
                    False,
                    trans_b=weight_trans,
                )
                if act == "sigmoid":
                    return torch.sigmoid(output)
                return output

        # bias 2d or None
        output = torch_mlu_ops.matmul(
            inputs,
            weight,
            None,
            bias,
            tmp_act,
            1.0,
            0.0 if bias is None else 1.0,
            self.fast_act,
            False,
            trans_b=weight_trans,
        )
        if act == "sigmoid":
            return torch.sigmoid(output)
        return output


class BatchDenseLayerModule(torch.nn.Module):
    def forward(self, inputs, weight, weight_trans, residual, bias, act):
        import torch_mlu_ops

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        if residual is not None:
            beta = 1.0
        else:
            beta = 0.0

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
