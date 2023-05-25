import torch
from tqdm import tqdm

from modules.config import parse_arguments
from modules.models import *
from modules.loader import DatasetLoader

args = parse_arguments()

# デバイスをGPUに設定
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの読み込み
extractor_model = FeatureExtractor(args)
trasformer_model = MyTransformer(args)
optimizer = torch.optim.Adam(trasformer_model.parameters(), lr=args.lr)

# データローダーの設定
train_dataset = DatasetLoader(args, phase="train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
val_dataset = DatasetLoader(args, phase="val")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, shuffle=False, drop_last=False)

# 学習ループ
for epoch in range(args.num_epochs):
    train_loop = tqdm(train_loader)
    for images, src_texts, tgt_texts in train_loop:
        with torch.no_grad():
            concated_embeddings = extractor_model(images, src_texts)
        loss = trasformer_model(concated_embeddings, tgt_texts)

        # 勾配の計算とパラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()