import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='プログラムの説明')
    # Model setting
    parser.add_argument('--image_model_name', type=str, default="microsoft/swinv2-base-patch4-window8-256", help='画像の特徴抽出モデル')
    parser.add_argument('--image_model_train', action='store_true', help='画像の特徴抽出モデルを学習するかどうか')
    parser.add_argument('--language_model_name', type=str, default='t5-large', choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'], help='言語の特徴抽出モデル')
    parser.add_argument('--transformer_model_name', type=str, default='t5-large', choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'], help='メインのモデル')
    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=128)
    # Training setting
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--batch_size', type=int, default=64, help='1GPUあたりのバッチサイズ')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='勾配の蓄積回数')
    parser.add_argument('--num_epochs', type=int, default=None, help='エポック数')
    parser.add_argument('--num_steps', type=int, default=None, help='学習ステップ数')
    parser.add_argument('--save_interval', type=int, default=None, help='モデルの保存間隔')
    # Dir setting
    parser.add_argument('--data_dir', type=str, default='/user/data/mscoco2017/', help='データのディレクトリ')
    parser.add_argument('--result_dir', type=str, default='results/', help='結果を保存するディレクトリ')
    args = parser.parse_args()
    return args