# 加藤研究室マルチモーダルモデル

## 環境構築
```console
conda env create -f=requirements.yml
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