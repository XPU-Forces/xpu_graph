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


def compare_inference(device, data_type, mod_golden, mod_target, bsz=80, input_dim=16):
    target_input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    golden_input = target_input.clone()

    with torch.inference_mode():
        if device == "cpu":
            rng_state = torch.random.get_rng_state()
            golden_output = mod_golden(golden_input)

            torch.random.set_rng_state(rng_state)
            target_output = mod_target(target_input)
        else:
            with torch.random.fork_rng(device_type=device):
                golden_output = mod_golden(target_input)

            target_output = mod_target(target_input)

    assert is_similar(target_input, golden_input)
    assert is_similar(golden_output, target_output)


def compare_inference_compile(device, data_type, ModCls, backend, bsz=80, input_dim=16):
    torch._dynamo.reset()
    golden = ModCls(input_dim).to(device=device, dtype=data_type).eval()
    compiled = ModCls(input_dim).to(device=device, dtype=data_type).eval()
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)
    compiled.load_state_dict(golden.state_dict())
    compare_inference(device, data_type, golden, compiled, bsz, input_dim)


def compare_training(device, data_type, mod_golden, mod_target, nsteps=10, bsz=8, input_dim=16):
    golden_input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    target_input = golden_input.clone()
    target_val = torch.randn((bsz, 1), device=device, dtype=data_type)

    optimizer_golden = torch.optim.AdamW(mod_golden.parameters())
    optimizer_target = torch.optim.AdamW(mod_target.parameters())
    optimizer_target.load_state_dict(optimizer_golden.state_dict())

    loss_fn = nn.MSELoss()

    for i in range(nsteps):
        if device == "cpu":
            rng_state = torch.random.get_rng_state()
            optimizer_golden.zero_grad()
            loss_golden = loss_fn(mod_golden(golden_input), target_val)
            loss_golden.backward()
            optimizer_golden.step()
            torch.random.set_rng_state(rng_state)
        else:
            with torch.random.fork_rng(device_type=device):
                optimizer_golden.zero_grad()
                loss_golden = loss_fn(mod_golden(golden_input), target_val)
                loss_golden.backward()
                optimizer_golden.step()

        optimizer_target.zero_grad()
        loss_target = loss_fn(mod_target(target_input), target_val)
        loss_target.backward()
        optimizer_target.step()

        print(f"Step: {i} golden: {loss_golden}, target: {loss_target}")

        assert is_similar(target_input, golden_input)
        assert is_similar(loss_golden, loss_target)


def compare_training_compile(device, data_type, ModCls, backend, nsteps=10, bsz=8, input_dim=16):
    torch._dynamo.reset()
    golden = ModCls(input_dim).to(device=device, dtype=data_type).train()
    compiled = ModCls(input_dim).to(device=device, dtype=data_type).train()
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=None)
    compiled.load_state_dict(golden.state_dict())
    compare_training(device, data_type, golden, compiled, nsteps, bsz, input_dim)
