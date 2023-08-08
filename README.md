# 加藤研究室マルチモーダルモデル

## 環境構築
```console
conda create -n mmm python=3.10 -y
conda activate mmm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

<br>

## 指定可能なパラメータ
`modules/config.py`を参照

### モデル設定

| parameter | 説明 | default |
| - | - | - |
| image_model_name | 画像の特徴抽出モデル | microsoft/swinv2-base-patch4-window8-256 |
| image_model_train | 画像の特徴抽出モデルを学習するかどうか | False |
| language_model_name | 言語の特徴抽出モデル | t5-large |
| ffn | 特徴抽出モデルの出力をFFNで変換するかどうか | False |
| image_vocab_size | 画像のボキャブラリサイズ | 16384 |
| loc_vocab_size | 位置のボキャブラリサイズ | 1000 |
| vae_ckpt_path | VAEの重みファイルのパス | checkpoints/vqgan.pt |
| max_source_length | 入力文の最大長 | 512 |
| max_target_length | 出力文の最大長 | 512 |

### 学習設定

| parameter | 説明 | default |
| - | - | - |
| pretrain | 事前学習かどうか | False |
| image_mask_ratio | 画像のマスク率 | 0.75 |
| seed | 乱数シード | 999 |
| lr | 学習率 | 0.001 |
| optimizer | Optimizer | AdamW |
| lr_scheduler | 学習率のスケジューラ | なし |
| batch_size | 1GPUあたりのバッチサイズ | 64 |
| accumulation_steps | 勾配の蓄積回数 | 1 |
| num_epochs | 学習エポック数 | なし |
| num_steps | 学習ステップ数 | なし |
| warmup_steps | 学習率のウォームアップステップ数 | なし |
| save_interval | モデルの保存間隔 | なし |
| datasets | 使用データセットの名前 | imagenet sun397 |

### ディレクトリ設定
| parameter | 説明 | default |
| - | - | - |
| root_dir | データセットのrootディレクトリ | /local/ |
| result_dir | 結果を保存するディレクトリ | results/ |

<br>

## RedCapsでのCaptionの自己教師ありPretrain（動作未確認）

15%の単語をマスクして、復元するように学習

```text
入力：I'm 18 <extra_id_0> And did this as my Senior Project . What does <extra_id_1> Think <extra_id_2>
出力：<extra_id_0> . <extra_id_1> Reddit <extra_id_2> ? <extra_id_3>
```

### SwinTransformerの重みを凍結して学習

```console
bash run_scripts/pretrain/train_only_transformer.sh
```

### SwinTransformerを含めて学習

```console
bash run_scripts/pretrain/train_with_swin.sh
```

<br>

## MSCOCOでのCaptionの学習

```text
入力：What does th image describe ?
出力：Caption
```

### SwinTransformerの重みを凍結して学習

```console
bash run_scripts/caption/train_only_transformer.sh
```

### SwinTransformerを含めて学習

```console
bash run_scripts/caption/train_with_swin.sh
```

<br>

## ImageNetでのクラス分類の学習

```text
入力：What does th image describe ?
出力：a photo of <image_label>
```

### SwinTransformerの重みを凍結して学習

```console
bash run_scripts/image_classify/train_only_transformer.sh
```

### SwinTransformerを含めて学習

```console
bash run_scripts/image_classify/train_with_swin.sh
```