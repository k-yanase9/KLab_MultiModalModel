import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='プログラムの説明')
    parser.add_argument('--id', default=None, help='実験ID')
    # Model setting
    parser.add_argument('--float_type', default='bfloat16', choices=['bfloat16', 'float16', 'float32'], help='浮動小数点の精度')
    parser.add_argument('-i', '--image_model_name', type=str, default="microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft", 
                        choices=[
                            "microsoft/resnet-50",
                            "microsoft/resnet-101",
                            "microsoft/resnet-152",
                            "microsoft/swinv2-base-patch4-window8-256",
                            "microsoft/swinv2-base-patch4-window16-256",
                            "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
                        ], help='画像の特徴抽出モデル')
    parser.add_argument('--image_model_train', action='store_true', help='画像の特徴抽出モデルを学習するかどうか')
    parser.add_argument('-l', '--language_model_name', type=str, default='google/flan-t5-large', 
                        choices=[
                            't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b',
                            'google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl',
                        ], help='言語の特徴抽出モデル')
    parser.add_argument('--language_model_train', action='store_true', help='言語の特徴抽出モデルの埋め込みを学習するかどうか')
    parser.add_argument('--ffn', action='store_false', help='特徴抽出モデルの出力をFFNで変換するかどうか')
    parser.add_argument('--transformer_d_model', type=int, default=768, help='メインTransformerのd_model')
    parser.add_argument('--transformer_d_ff', type=int, default=3072, help='メインTransformerのd_ff')
    parser.add_argument('--transformer_d_kv', type=int, default=64, help='メインTransformerのd_kv')
    parser.add_argument('--transformer_num_heads', type=int, default=12, help='メインTransformerのヘッド数')
    parser.add_argument('--transformer_num_layers', type=int, default=2, help='メインTransformerの層数')
    parser.add_argument('--transformer_num_decoder_layers', type=int, default=12, help='メインTransformerのデコーダーの層数')
    parser.add_argument('--transformer_model_init', type=str, default='random', choices=['random', 'pretrain', 'few', 't5-base', 'google/flan-t5-base'], help='メインTransformerの重みの初期化方法')
    parser.add_argument('--additional_vocab_size', type=int, default=10000, help='予備のボキャブラリサイズ', choices=[0, 1000, 10000, 16384])
    parser.add_argument('--loc_vocab_size', type=int, default=1600, help='位置のボキャブラリサイズ', choices=[1000, 1600])
    parser.add_argument('--vae_ckpt_path', type=str, default='', choices=['', 'checkpoints/vqgan.pt'], help='VAEの重みファイルのパス')
    parser.add_argument('--max_source_length', type=int, default=256, help='入力文の最大長')
    parser.add_argument('--max_target_length', type=int, default=256, help='出力文の最大長')
    # Training setting
    parser.add_argument('--multinode', action='store_true', help='マルチノードで学習するかどうか')
    parser.add_argument('--stage', type=str, default='train', choices=['pretrain', 'train', 'classify', 'finetune','zeroshot'], help='事前学習か学習か分類か')
    parser.add_argument('--seed', type=int, default=999, help='乱数シード')
    parser.add_argument('--loss', type=str, default='CrossEntropy', choices=['CrossEntropy', 'FocalLoss'], help='損失関数')
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='', choices=['', 'Lambda', 'Cosine', 'Exponential', 'Step', 'MultiStep', 'LinearWarmup', 'CosineWarmup'], help='学習率のスケジューラ')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='1GPUあたりのバッチサイズ')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='勾配の蓄積回数')
    parser.add_argument('--start_epoch', type=int, default=1, help='初期エポック')
    parser.add_argument('--stop_epoch', type=int, default=0, help="停止エポック")
    parser.add_argument('--num_epochs', type=int, default=None, help='学習エポック数')
    parser.add_argument('--num_steps', type=int, default=None, help='学習ステップ数')
    parser.add_argument('--warmup_rate', type=float, default=0.01, help='ウォームアップの割合')
    parser.add_argument('--save_interval', type=int, default=None, help='モデルの保存間隔')
    parser.add_argument('--datasets', nargs='+', default=['imagenet', 'sun397'], choices=[
        'imagenet', 'imagenet21k', 'inaturalist', 'places365', 'sun397',  
        'redcaps', 'cc3m', 'cc12m', 'mscoco', 'grit20m_rcap', 'grit20m_refexp',
        'vcr', 'vqa2', 'imSitu', 'tdiuc', 'visual7w_vqa', 'visual7w_gvqa', 
        'openimage_cat', 'openimage_det', 'openimage_loc', 'openimage_rel', 
        'objects365_cat', 'objects365_det', 'objects365_loc', 
        'vg_cat', 'vg_det', 'vg_loc', 'vg_rel', 'vg_vqa', 'vg_rcap', 'vg_refexp', 
        'hico', 'vcoco', 'icdar_loc', 'icdar_read', 'deepfashion2_cat', 'deepfashion2_loc', 'deepfashion2_det',
        'all',
        'caption', 'relation', 'rcap', 'refexp', 'det', 'cat', 'loc', 'vqa', 'gvqa', 'classify'
        ], help='使用データセットの名前')
    parser.add_argument('--is_tgt_id', action='store_true', help='ターゲットIDを使用するかどうか')
    parser.add_argument('--uncalc_val', action='store_true', help='検証を行わない')
    # Dir setting
    parser.add_argument('--root_dir', type=str, default='/local/', help='データのディレクトリ')
    parser.add_argument('--result_dir', type=str, default='results/', help='結果を保存するディレクトリ')
    args = parser.parse_args()
    if args.stop_epoch == 0:
        args.stop_epoch = args.num_epochs
    return args