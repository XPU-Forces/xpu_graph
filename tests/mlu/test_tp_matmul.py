import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

import xpu_graph
from tests.mlu.test_dist_train_utils import (
    MainModel,
    cleanup,
    get_dp_dataloader,
    set_dist_env,
    train_setup,
)
from xpu_graph.config import OptLevel


def train(rank, world_size, do_compile, return_queue, ModCls, model_path):
    device_mesh = train_setup(rank, world_size)
    model = ModCls()
    model.load_state_dict(torch.load(model_path))
    if do_compile:
        xpu_graph_backend = xpu_graph.mlu_compiler(is_training=True, freeze=False, opt_level=OptLevel.level2)
        model = torch.compile(model, backend=xpu_graph_backend, dynamic=False)
    model.mlu(rank)
    model.inner = parallelize_module(
        model.inner,
        device_mesh,
        parallelize_plan={
            "inner.0": ColwiseParallel(input_layouts=Shard(0), output_layouts=Shard(1)),
            "inner.2": RowwiseParallel(input_layouts=Shard(1), output_layouts=Shard(0)),
        },
    )
    print(model.inner)
    print({name: param.shape for name, param in model.inner.named_parameters()})

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    dataloader = get_dp_dataloader(rank, world_size)

    final_loss = 0
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.mlu(rank), target.mlu(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

            final_loss = loss

    return_queue.put((rank, final_loss.item()))
    cleanup()


def tp_test(ModCls, model_path="tp_model.pth"):
    set_dist_env()
    mp.set_start_method("spawn", force=True)
    world_size = torch.mlu.device_count()
    return_queue1 = mp.Queue()
    return_queue2 = mp.Queue()
    model = ModCls()
    torch.save(model.state_dict(), model_path)

    do_compile = 0
    torch.multiprocessing.spawn(
        train,
        args=(world_size, do_compile, return_queue1, ModCls, model_path),
        nprocs=world_size,
        join=True,
    )
    results1 = {}
    for _ in range(world_size):
        rank, loss = return_queue1.get()
        results1[rank] = loss

    do_compile = 1
    torch.multiprocessing.spawn(
        train,
        args=(world_size, do_compile, return_queue2, ModCls, model_path),
        nprocs=world_size,
        join=True,
    )
    results2 = {}
    for _ in range(world_size):
        rank, loss = return_queue2.get()
        results2[rank] = loss

    for i in range(world_size):
        assert abs(results1[i] - results2[i]) < 0.01


class TestTP:
    @pytest.mark.parametrize(
        "PatternModel",
        [
            MainModel,
        ],
    )
    def test_tp(self, tmp_path, PatternModel):
        tp_test(PatternModel, tmp_path / "tp_model.pth")


if __name__ == "__main__":
    tp_test(MainModel)
