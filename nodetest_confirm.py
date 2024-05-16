import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class RandomDataset(Dataset):
    def __init__(self, input_size, length):
        self.len = length
        self.data = torch.randn(length, input_size)
        self.targets = torch.randn(length, 1)  # ターゲットをデータセットに含める

    def __getitem__(self, index):
        return self.data[index], self.targets[index]  # データとターゲットのタプルを返す

    def __len__(self):
        return self.len

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        return self.fc(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12345')

    hostname = socket.gethostname()
    
    # ホスト名に基づいてNCCLのネットワークインターフェース名を設定
    if hostname == "a100-40gbx4-01":
        nccl_ifname = "eno1"
    elif hostname == "a100-40gbx4-02":
        nccl_ifname = "enp193s0f0"
    else:
        nccl_ifname = "default"

    os.environ["NCCL_SOCKET_IFNAME"] = nccl_ifname
    # print(f"Rank {rank}/{world_size}, Master Address: {os.environ['MASTER_ADDR']}, Master Port: {os.environ['MASTER_PORT']}, NCCL_IFNAME: {os.environ['NCCL_SOCKET_IFNAME']}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    local_rank = int(os.getenv('SLURM_LOCALID', '0'))
    torch.cuda.set_device(local_rank)

def train(rank, world_size, epochs):
    setup(rank, world_size)
    model = SimpleModel().cuda()
    ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])

    dataset = RandomDataset(20, 100)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for data, target in loader:
            optimizer.zero_grad()
            output = ddp_model(data.cuda())
            loss = criterion(output, target.cuda())
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f'Epoch {epoch}: Loss {loss.item()}')
    return ddp_model  # トレーニングされたモデルを返す

def check_model_consistency(ddp_model, rank, world_size):
    # モデルのパラメータを全ランクで収集
    for param in ddp_model.module.parameters():  # ddp_model.module で元のモデルのパラメータにアクセス
        gathered_params = [param.clone() for _ in range(world_size)]
        dist.all_gather(gathered_params, param)

        if rank == 0:
            base_param = gathered_params[0]
            for other_param in gathered_params[1:]:
                assert torch.allclose(base_param, other_param), "Models diverged!"

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    ddp_model = train(rank, world_size, 10)
    check_model_consistency(ddp_model, rank, world_size)