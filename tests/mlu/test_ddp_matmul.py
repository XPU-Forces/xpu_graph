import pytest

from tests.mlu.test_dist_train_utils import *


def train(rank, world_size, do_compile, return_queue, model_path):
    train_setup(rank, world_size)
    model = MatMulModel(10)
    model.load_state_dict(torch.load(model_path))
    if do_compile:
        xpu_graph_backend = xpu_graph.mlu_compiler(is_training=True, freeze=False, opt_level=OptLevel.level2)
        model = torch.compile(model, backend=xpu_graph_backend, dynamic=False)
    model.mlu(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = RandomDataset(size=10, length=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    final_loss = 0
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.mlu(rank), target.mlu(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch [{epoch}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

            final_loss = loss

    return_queue.put((rank, final_loss.item()))
    cleanup()


def ddp_test(ModCls, model_path="ddp_model.pth"):
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
        args=(world_size, do_compile, return_queue1, model_path),
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
        args=(world_size, do_compile, return_queue2, model_path),
        nprocs=world_size,
        join=True,
    )
    results2 = {}
    for _ in range(world_size):
        rank, loss = return_queue2.get()
        results2[rank] = loss

    for i in range(world_size):
        assert abs(results1[i] - results2[i]) < 0.01


class TestDDP:
    @pytest.mark.parametrize(
        "pattern_func",
        [
            MatMulModel,
        ],
    )
    def test_rmsnorm_patterns(self, tmp_path, pattern_func):
        ddp_test(pattern_func, tmp_path / "ddp_model.pth")


if __name__ == "__main__":
    ddp_test(MatMulModel)
