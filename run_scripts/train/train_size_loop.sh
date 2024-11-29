# 1ノード8GPU RTX6000Ada

# デフォルト値の設定
DEFAULT_ENC=2
DEFAULT_DEC=12
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCH=50
DEFAULT_DATASET="all"
lr=1e-4

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

torchrun --nnodes=1 --nproc_per_node=8 multi_task_train.py \
        --transformer_num_layers $enc \
        --transformer_num_decoder_layers $dec \
        --transformer_model_init random \
        --stage train \
        --loss CrossEntropy \
        --lr $lr \
        --lr_scheduler LinearWarmup \
        -b $batch_size \
        --start_epoch 1 \
        --num_epochs $epoch \
        --warmup_rate 0.01 \
        --datasets $dataset \
        --root_dir /home/data/ \
        --save_interval 2 \
        --result_dir "results/enc${enc}_dec${dec}/alltask_random"