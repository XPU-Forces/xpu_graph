import torch
import torch.nn as nn

from xpu_graph.test_utils import is_similar


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
        x = torch.abs(x) + 1
        y = x.clone()
        x.add_(1)
        z = x * y
        return z.sum(dim=-1)


class ConstantInplaceModel(nn.Module):
    def __init__(self, input_dim):
        super(ConstantInplaceModel, self).__init__()
        self.fc = nn.Linear(input_dim, 16)

    def forward(self, x):
        x = self.fc(x)
        indices = x.sum(dim=-1).nonzero().squeeze(-1)
        indices = indices[max(indices.shape[0] // 2, 1) :]
        y = x[indices].sum(-1)
        max_len = indices.max() + 1
        zeros = torch.zeros(max_len, dtype=y.dtype, device=y.device)
        zeros.scatter_(0, indices, y)
        result = torch.cat([zeros, torch.zeros(x.shape[0] - max_len, dtype=zeros.dtype, device=zeros.device)], dim=0)
        return result


class ConstantInplaceModelV2(nn.Module):
    def __init__(self, input_dim):
        super(ConstantInplaceModelV2, self).__init__()
        self.fc = nn.Linear(input_dim, 16)

    def forward(self, x):
        x = self.fc(x)
        indices = x.sum(dim=-1).nonzero().squeeze(-1)
        indices = indices[max(indices.shape[0] // 2, 1) :]
        y = x[indices].sum(-1)
        max_len = indices.max() + 1
        zeros = torch.zeros(max_len, dtype=y.dtype, device=y.device)
        zeros = zeros.scatter_(0, indices, y)
        result = torch.cat([zeros, torch.zeros(x.shape[0] - max_len, dtype=zeros.dtype, device=zeros.device)], dim=0)
        return result


class DropoutModel(nn.Module):
    def __init__(self, input_dim):
        super(DropoutModel, self).__init__()
        self.fc = nn.Linear(input_dim, 16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x


all_models = [SimpleModel, SliceCatModel, InplaceModel, ConstantInplaceModel, DropoutModel]


def compare_inference_between(golden, compiled, example_inputs):
    compiled_input = example_inputs.clone()
    golden_input = example_inputs.clone()
    with torch.inference_mode():
        with torch.random.fork_rng(device_type=torch._utils._get_available_device_type()):
            output_golden = golden(golden_input)

        compiled_output = compiled(compiled_input)

    assert is_similar(compiled_input, golden_input)
    assert is_similar(compiled_output, output_golden)


def compare_inference(device, data_type, ModCls, backend, bsz=80, input_dim=16, dynamic=True, state_dict=None):
    golden = ModCls(input_dim).to(device=device, dtype=data_type).eval()
    compiled = ModCls(input_dim).to(device=device, dtype=data_type).eval()

    if state_dict is not None:
        compiled.load_state_dict(state_dict)
        golden.load_state_dict(state_dict)
    else:
        compiled.load_state_dict(golden.state_dict())

    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=dynamic)
    compiled_input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    compare_inference_between(golden, compiled, compiled_input)


def compare_training_between(golden, compiled, example_inputs, target):
    compiled.load_state_dict(golden.state_dict())

    nsteps = example_inputs.shape[0]
    optimizer_golden = torch.optim.AdamW(golden.parameters())
    optimizer_compiled = torch.optim.AdamW(compiled.parameters())
    optimizer_compiled.load_state_dict(optimizer_golden.state_dict())
    print(optimizer_golden.state_dict(), optimizer_compiled.state_dict())

    loss_fn = nn.MSELoss()

    for i in range(nsteps):
        compiled_input = example_inputs[i].clone()
        golden_input = example_inputs[i].clone()
        optimizer_golden.zero_grad()
        with torch.random.fork_rng(device_type=torch._utils._get_available_device_type()):
            loss_golden = loss_fn(golden(golden_input), target[i])
            loss_golden.backward()

        optimizer_golden.step()

        optimizer_compiled.zero_grad()
        loss_compiled = loss_fn(compiled(compiled_input), target[i])
        loss_compiled.backward()
        optimizer_compiled.step()

        print(f"Step: {i} golden: {loss_golden}, compiled: {loss_compiled}")

        assert is_similar(compiled_input, golden_input)
        assert is_similar(loss_golden, loss_compiled)


def compare_training(device, data_type, ModCls, backend, nsteps=10, bsz=8, input_dim=16, dynamic=True):
    golden = ModCls(input_dim).to(device=device, dtype=data_type).train()
    compiled = ModCls(input_dim).to(device=device, dtype=data_type).train()

    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=dynamic)

    example_inputs = torch.randn((nsteps, bsz, input_dim), device=device, dtype=data_type)
    example_target = torch.randn((nsteps, bsz, 1), device=device, dtype=data_type)

    compare_training_between(golden, compiled, example_inputs, example_target)
