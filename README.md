# 加藤研究室マルチモーダルモデル

## 環境構築
```console
conda create -n mmm python=3.10 -y
conda activate mmm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## RedCapsでのCaptionの自己教師あり事前学習（動作未確認）

15%の単語をマスクして、復元するように学習

入力：I'm 18 <extra_id_0> And did this as my Senior Project . What does <extra_id_1> Think <extra_id_2>

出力：<extra_id_0> . <extra_id_1> Reddit <extra_id_2> ? <extra_id_3>

```console

### SwinTransformerの重みを凍結して学習

```console
bash run_scripts/pretrain/train_only_transformer.sh
```

### SwinTransformerを含めて学習

```console
bash run_scripts/pretrain/train_with_swin.sh
```

## MSCOCOでのCaptionの学習

入力：What does th image describe ?

出力：Caption

### SwinTransformerの重みを凍結して学習

```console
bash run_scripts/caption/train_only_transformer.sh
```

### SwinTransformerを含めて学習

```console
bash run_scripts/caption/train_with_swin.sh
```