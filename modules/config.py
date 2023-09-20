import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='プログラムの説明')
    # Model setting
    parser.add_argument('-i', '--image_model_name', type=str, default="microsoft/swinv2-base-patch4-window8-256", 
                        choices=[
                            "microsoft/resnet-50",
                            "microsoft/resnet-101",
                            "microsoft/resnet-152",
                            "microsoft/swinv2-base-patch4-window8-256",
                            "microsoft/swinv2-base-patch4-window16-256",
                            "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
                        ], help='画像の特徴抽出モデル')
    parser.add_argument('--image_model_train', action='store_true', help='画像の特徴抽出モデルを学習するかどうか')
    parser.add_argument('-l', '--language_model_name', type=str, default='google/flan-t5-base', 
                        choices=[
                            't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b',
                            'google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl',
                        ], help='言語の特徴抽出モデル')
    parser.add_argument('--ffn', action='store_true', help='特徴抽出モデルの出力をFFNで変換するかどうか')
    parser.add_argument('--transformer_d_model', type=int, default=512, help='メインTransformerのd_model')
    parser.add_argument('--transformer_d_ff', type=int, default=2048, help='メインTransformerのd_ff')
    parser.add_argument('--transformer_d_kv', type=int, default=64, help='メインTransformerのd_kv')
    parser.add_argument('--transformer_num_heads', type=int, default=2, help='メインTransformerのヘッド数')
    parser.add_argument('--transformer_num_layers', type=int, default=8, help='メインTransformerの層数')
    parser.add_argument('--transformer_num_decoder_layers', type=int, default=8, help='メインTransformerのデコーダーの層数')
    parser.add_argument('--image_vocab_size', type=int, default=0, help='画像のボキャブラリサイズ', choices=[0, 16384])
    parser.add_argument('--loc_vocab_size', type=int, default=1600, help='位置のボキャブラリサイズ', choices=[1000, 1600])
    parser.add_argument('--vae_ckpt_path', type=str, default='', choices=['', 'checkpoints/vqgan.pt'], help='VAEの重みファイルのパス')
    parser.add_argument('--max_source_length', type=int, default=256, help='入力文の最大長')
    parser.add_argument('--max_target_length', type=int, default=256, help='出力文の最大長')
    # Training setting
    parser.add_argument('--pretrain', action='store_true', help='事前学習かどうか')
    parser.add_argument('--seed', type=int, default=999, help='乱数シード')
    parser.add_argument('--loss', type=str, default='FocalLoss', choices=['CrossEntropy', 'FocalLoss'], help='損失関数')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='', choices=['', 'LambdaLR', 'CosineAnnealingLR', 'ExponentialLR', 'StepLR', 'MultiStepLR', 'LinearWarmup', 'CosineWarmup'], help='学習率のスケジューラ')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='1GPUあたりのバッチサイズ')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='勾配の蓄積回数')
    parser.add_argument('--start_epoch', type=int, default=1, help='初期エポック')
    parser.add_argument('--num_epochs', type=int, default=None, help='学習エポック数')
    parser.add_argument('--num_steps', type=int, default=None, help='学習ステップ数')
    parser.add_argument('--warmup_steps', type=int, default=None, help='学習率を上げるステップ数')
    parser.add_argument('--save_interval', type=int, default=None, help='モデルの保存間隔')
    parser.add_argument('--datasets', nargs='+', default=['imagenet', 'sun397'], choices=['redcaps', 'imagenet', 'imagenet_21k', 'places365', 'inaturalist', 'cc3m', 'cc12m', 'sun397', 'mscoco', 'vcr', 'vqa2', 'imsitu', 'imagenet', 'openimage'], help='使用データセットの名前')
    # Dir setting
    parser.add_argument('--root_dir', type=str, default='/user/data/', help='データのディレクトリ')
    parser.add_argument('--result_dir', type=str, default='results/', help='結果を保存するディレクトリ')
    args = parser.parse_args()
    return args