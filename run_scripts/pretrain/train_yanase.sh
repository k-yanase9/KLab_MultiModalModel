# 1ノード8GPU RTX6000Ada

# デフォルト値の設定
DEFAULT_ENC=2
DEFAULT_DEC=12
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCH=20
DEFAULT_DATASET="cc3m cc12m imagenet imagenet21k places365 redcaps sun397"

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --enc)
            enc="$2"
            shift 2
            ;;
        --dec)
            dec="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --epoch)
            epoch="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# デフォルト値の適用
enc=${enc:-$DEFAULT_ENC}
dec=${dec:-$DEFAULT_DEC}
batch_size=${batch_size:-$DEFAULT_BATCH_SIZE}
epoch=${epoch:-$DEFAULT_EPOCH}
dataset=${dataset:-"$DEFAULT_DATASET"}

# 値の表示（オプション）
echo "Using configuration:"
echo "Encoder layers: $enc"
echo "Decoder layers: $dec"
echo "Batch size: $batch_size"
echo "Epochs: $epoch"
echo "Dataset: $dataset"

# トレーニングコマンドの実行
torchrun --nnodes=1 --nproc_per_node=8 train.py \
        --transformer_num_layers "$enc" \
        --transformer_num_decoder_layers "$dec" \
        --stage pretrain \
        --loss CrossEntropy \
        --lr 1e-4 \
        --lr_scheduler LinearWarmup \
        -b "$batch_size" \
        --start_epoch 1 \
        --num_epochs "$epoch" \
        --warmup_rate 0.001 \
        --datasets "$dataset" \
        --root_dir /home/data/ \
        --save_interval 1 \
        --result_dir "results/enc${enc}_dec${dec}/pretrain"