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
| float_type | floatの精度 | bfloat16 |
| image_model_name | 画像の特徴抽出モデル | microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft |
| image_model_train | 画像の特徴抽出モデルを学習するかどうか | False |
| language_model_name | 言語の特徴抽出モデル | google/flan-t5-large |
| ffn | 特徴抽出モデルの出力をFFNで変換するかどうか | True |
| transformer_d_model | メインTransformerのd_model | 768 |
| transformer_d_ff | メインTransformerのd_ff | 3072 |
| transformer_d_kv | メインTransformerのd_kv | 64 |
| transformer_num_heads | メインTransformerのヘッド数 | 12 |
| transformer_num_layers | メインTransformerの層数 | 12 |
| transformer_num_decoder_layers | メインTransformerのデコーダの層数 | 12 |
| additional_vocab_size | 予備のボキャブラリサイズ | 10000 |
| loc_vocab_size | 位置のボキャブラリサイズ | 1600 |
| max_source_length | 入力文の最大長 | 512 |
| max_target_length | 出力文の最大長 | 512 |

### 学習設定

| parameter | 説明 | default |
| - | - | - |
| multinode | マルチノード学習を行うかどうか | False |
| phase | 事前学習か学習か分類か | train |
| seed | 乱数シード | 999 |
| loss | 損失関数 | CrossEntropy |
| lr | 学習率 | 0.01 |
| optimizer | Optimizer | AdamW |
| lr_scheduler | 学習率のスケジューラ | なし |
| batch_size | 1GPUあたりのバッチサイズ | 64 |
| accumulation_steps | 勾配の蓄積回数 | 1 |
| start_epoch | 初期エポック | 1 |
| num_epochs | 学習エポック数 | なし |
| num_steps | 学習ステップ数 | なし |
| warmup_rate | ウォームアップの割合 | 0.01 |
| save_interval | モデルの保存間隔 | なし |
| datasets | 使用データセットの名前 | imagenet sun397 |

### ディレクトリ設定
| parameter | 説明 | default |
| - | - | - |
| root_dir | データセットのrootディレクトリ | /local/ |
| result_dir | 結果を保存するディレクトリ | results/ |

<br>

## 自己教師ありPretrain（CC12Mなど）

15%の単語をマスクして、復元するように学習

```text
入力：I'm 18 <extra_id_0> And did this as my Senior Project . What does <extra_id_1> Think <extra_id_2>
出力：<extra_id_0> . <extra_id_1> Reddit <extra_id_2> ? <extra_id_3>
```

### 学習

```console
bash run_scripts/pretrain/train.sh
```

<br>

## Captionの学習

```text
入力：What does th image describe ?
出力：Caption
```

### 学習

```console
bash run_scripts/caption/train.sh
```

<br>

## SUN397でのクラス分類の学習

```text
入力：What does th image describe ?
出力：a photo of <image_label>
```

### s学習

```console
bash run_scripts/image_classify/train.sh
```
