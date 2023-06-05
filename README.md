# 加藤研究室マルチモーダルモデル

## 環境構築
```console
conda create -n mmm python=3.10 -y
conda activate mmm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 実行

### SwinTransformerの重みを凍結して学習

```console
bash run_scripts/caption/train_only_transformer.sh
```

### SwinTransformerを含めて学習

```console
bash run_scripts/caption/train_with_swin.sh
```