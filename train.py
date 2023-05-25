import torch
from torch.nn.parallel import DataParallel
from tqdm import tqdm

from modules.config import parse_arguments
from modules.models import *
from modules.loader import DatasetLoader

args = parse_arguments()

# デバイスをGPUに設定
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの読み込み
model = MyModel(args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# データローダーの設定
train_dataset = DatasetLoader(args, phase="train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
val_dataset = DatasetLoader(args, phase="val")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)

for epoch in range(args.num_epochs):
    # 学習ループ
    train_loop = tqdm(train_loader)
    for images, src_texts, tgt_texts in train_loop:
        loss = model(images, src_texts, tgt_texts)

        # 勾配の計算とパラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 検証ループ
    val_losses = []
    for images, src_texts, tgt_texts in val_loader:
        with torch.no_grad():
            loss = model(images, src_texts, tgt_texts)
            val_losses.append(loss)

    print(f'{epoch+1}: {torch.mean(val_losses)}')