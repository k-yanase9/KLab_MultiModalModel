import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from ex_module import ExModel, MyChainDataset, MyDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


# ユーザー関数定義
def parse_arguments():
    parser = argparse.ArgumentParser(description='プログラムの説明')
    # Model setting
    parser.add_argument(
        '-i',
        '--image_model_name',
        type=str,
        default="microsoft/swinv2-base-patch4-window8-256",
        choices=[
            "microsoft/resnet-50",
            "microsoft/resnet-101",
            "microsoft/resnet-152",
            "microsoft/swinv2-base-patch4-window8-256",
            "microsoft/swinv2-base-patch4-window16-256",
            "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
        ],
        help='画像の特徴抽出モデル',
    )
    parser.add_argument('--image_model_train', action='store_true', help='画像の特徴抽出モデルを学習するかどうか')
    parser.add_argument(
        '-l',
        '--language_model_name',
        type=str,
        default='google/flan-t5-base',
        choices=[
            't5-small',
            't5-base',
            't5-large',
            't5-3b',
            't5-11b',
            'google/flan-t5-small',
            'google/flan-t5-base',
            'google/flan-t5-large',
            'google/flan-t5-xl',
            'google/flan-t5-xxl',
        ],
        help='言語の特徴抽出モデル',
    )
    parser.add_argument('--ffn', action='store_true', help='特徴抽出モデルの出力をFFNで変換するかどうか')
    parser.add_argument('--transformer_d_model', type=int, default=512, help='メインTransformerのd_model')
    parser.add_argument('--transformer_d_ff', type=int, default=2048, help='メインTransformerのd_ff')
    parser.add_argument('--transformer_d_kv', type=int, default=64, help='メインTransformerのd_kv')
    parser.add_argument('--transformer_num_heads', type=int, default=2, help='メインTransformerのヘッド数')
    parser.add_argument('--transformer_num_layers', type=int, default=8, help='メインTransformerの層数')
    parser.add_argument('--transformer_num_decoder_layers', type=int, default=8, help='メインTransformerのデコーダーの層数')
    parser.add_argument('--image_vocab_size', type=int, default=16384, help='画像のボキャブラリサイズ')
    parser.add_argument('--loc_vocab_size', type=int, default=1000, help='位置のボキャブラリサイズ')
    parser.add_argument('--vae_ckpt_path', type=str, default='checkpoints/vqgan.pt', help='VAEの重みファイルのパス')
    parser.add_argument('--max_source_length', type=int, default=256, help='入力文の最大長')
    parser.add_argument('--max_target_length', type=int, default=256, help='出力文の最大長')
    # Training setting
    parser.add_argument('--pretrain', action='store_true', help='事前学習かどうか')
    parser.add_argument('--seed', type=int, default=999, help='乱数シード')
    parser.add_argument('--loss', type=str, default='FocalLoss', choices=['CrossEntropy', 'FocalLoss'], help='損失関数')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='Optimizer')
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='',
        choices=['', 'LambdaLR', 'CosineAnnealingLR', 'ExponentialLR', 'StepLR', 'MultiStepLR', 'LinearWarmup', 'CosineWarmup'],
        help='学習率のスケジューラ',
    )
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='1GPUあたりのバッチサイズ')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='勾配の蓄積回数')
    parser.add_argument('--num_epochs', type=int, default=None, help='学習エポック数')
    parser.add_argument('--num_steps', type=int, default=None, help='学習ステップ数')
    parser.add_argument('--warmup_steps', type=int, default=None, help='学習率を上げるステップ数')
    parser.add_argument('--save_interval', type=int, default=None, help='モデルの保存間隔')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['imagenet', 'sun397'],
        choices=['redcaps', 'imagenet', 'imagenet_21k', 'places365', 'inaturalist', 'cc3m', 'cc12m', 'sun397', 'mscoco', 'vcr', 'vqa2', 'imsitu', 'imagenet'],
        help='使用データセットの名前',
    )
    # Dir setting
    parser.add_argument('--root_dir', type=str, default='/user/data/', help='データのディレクトリ')
    parser.add_argument('--result_dir', type=str, default='results/', help='結果を保存するディレクトリ')
    args = parser.parse_args()
    return args


def get_logger(args, world_rank,log_name='train.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')

    # ログのコンソール出力の設定
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # ログのファイル出力先を設定
    fh = logging.FileHandler(os.path.join(args.result_dir,f"rank_{world_rank}", log_name), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Options: {args.__dict__}")

    return logger


from torch.optim import *


def get_optimizer(model, args):
    if args.optimizer == 'Adam':
        return Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        return AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise NotImplementedError


def train(_,world_rank,local_rank,world_size,port_num,host_list_file):
    with open(host_list_file) as f:
        host = f.readlines()
    host[0] = host[0].rstrip("\n")
    dist_url = "tcp://" + host[0] + ":" + str(port_num)
    
    dist.init_process_group(backend="nccl",init_method=dist_url,rank=world_rank,world_size=world_size)

    args = parse_arguments()
    args.gpu_nums = world_size #torch.cuda.device_count()  # GPU数
    #device_id = rank % args.gpu_nums
    device_id = torch.device("cuda:{}".format(local_rank))
    

    if not os.path.exists(os.path.join(args.result_dir,f"rank_{world_rank}")):
        os.makedirs(os.path.join(args.result_dir,f"rank_{world_rank}"), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # if rank == 0:
    logger = get_logger(args,world_rank)
    logger.info("make_logger")

    # create model
    model = ExModel(args).to(device_id)
    model = DDP(model, device_ids=[device_id],output_device=device_id)

    optimizer = get_optimizer(model, args)
    criterion = torch.nn.MSELoss()
    # scheduler = get_scheduler(args, optimizer)

    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # src_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True)
    # tgt_tokenizer = AutoTokenizer.from_pretrained(args.language_model_name, model_max_length=256, use_fast=True, extra_ids=0, additional_special_tokens =[f"<extra_id_{i}>" for i in range(100)] + [f"<loc_{i}>" for i in range(args.loc_vocab_size)] + [f"<img_{i}>" for i in range(args.image_vocab_size)])

    # データの設定
    # train_dataset, val_dataset = get_data(args, src_tokenizer, tgt_tokenizer)
    # train_loader = get_distributed_dataloader(args, train_dataset, shuffle=True)
    # val_loader = get_distributed_dataloader(args, val_dataset, shuffle=False)

    # if args.num_epochs is None:
    #     args.num_epochs = int(args.num_steps / len(train_loader)) + 1

    task_num = 3  # taskの数=datasetの数
    batch_size_list = [5, 50, 10]  # 1 : 10 : 2
    batch_size_list = [15, 50]
    # 60,200
    # 300,1000
    # データの数は task1: 100,task2: 1100,task3: 300
    # step当たりのbatch_sizeはgpu数は4で4xbatch_size[20,200,40]
    # step=5で[100.1000,200]となりtask1のデータが終わる

    # データセット読み込み
    datasets = [MyDataset(f"dataset_{i+1}.csv") for i in range(task_num)]
    # dataset_1 = MyDataset("dataset_1.csv")
    # dataset_2 = MyDataset("dataset_2.csv")
    # dataset_3 = MyDataset("dataset_3.csv")

    datasets = [MyChainDataset([datasets[0], datasets[2]]), datasets[1]]

    distributed_samplers = [
        torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=False, drop_last=True)
        for dataset in datasets
    ]
    # distributed_sampler_1 = torch.utils.data.distributed.DistributedSampler(dataset_1, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    # distributed_sampler_2 = torch.utils.data.distributed.DistributedSampler(dataset_2, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    # distributed_sampler_3 = torch.utils.data.distributed.DistributedSampler(dataset_3, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    loaders = [
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=distributed_sampler, num_workers=4, pin_memory=True)
        for dataset, batch_size, distributed_sampler in zip(datasets, batch_size_list, distributed_samplers)
    ]
    min_step = min([len(loader) for loader in loaders])
    max_step = max([len(loader) for loader in loaders])

    # loader_1 = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=distributed_sampler_1)
    # loader_2 = torch.utils.data.DataLoader(dataset_2, batch_size=100, sampler=distributed_sampler_2)
    # loader_3 = torch.utils.data.DataLoader(dataset_3, batch_size=20, sampler=distributed_sampler_3)

    # steps = 0
    # min_val_loss = 100
    # loss_counter = LossCounter()
    prefix_text = f"rank:{world_rank} || "
    logger.info(f"{prefix_text}start training")
    logger.info(f"min_step: {min_step}, max_step: {max_step}")
    for epoch in range(1, args.num_epochs + 1):
        # # 学習ループ
        # image_mask_ratio = 0.0
        # if args.image_model_train:
        #     model.module.image_model.train()
        # model.module.transformer.train()
        # train_loss = torch.tensor(0.0).to(device_id)
        # train_count = torch.tensor(0).to(device_id)

        # pbar = tqdm(total=int(np.ceil(min_step / args.accumulation_steps)), desc=f'Train (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))

        # すべてのiteratorを初期化する
        iterators = [iter(loader) for loader in loaders]
        for step in range(1, min_step + 1):
            prefix_text = f"epoch:{epoch} step:{step} rank:{world_rank} || "

            # iteratorはrankごとに初期化することができる、他のrankのiteratorは初期化されない
            # rank=0のGPUでstep == 3の時にiteratorを初期化する
            if world_rank == 0 and epoch == 2 and step == 3:
                iterator_index = 1
                logger.info(f"{prefix_text} initialize rank:{world_rank} iterator {iterator_index}")
                iterators[iterator_index] = iter(loaders[iterator_index])

            # すべてのiteratorからデータを取得し結合する
            image_list = []
            in_text_list = []
            out_text_list = []
            for iterator in iterators:
                image, in_text, out_text = next(iterator)
                image_list.append(image)
                in_text_list.append(in_text)
                out_text_list.append(out_text)
            image = torch.cat(image_list).to(device_id)
            in_text = torch.cat(in_text_list).to(device_id)
            out_text = torch.cat(out_text_list).to(device_id)

            # データをlogで確認
            sample_image_list = []
            sample_in_list = []
            sample_out_list = []
            sample_num = 2  # 1バッチの中から何個サンプルするか
            in_batch_index = 0  # バッチ中のindex
            for batch_size in batch_size_list:
                sample_image_list.append(image[in_batch_index : in_batch_index + sample_num].flatten())
                sample_in_list.append(in_text[in_batch_index : in_batch_index + sample_num].flatten())
                sample_out_list.append(out_text[in_batch_index : in_batch_index + sample_num].flatten())
                in_batch_index += batch_size
            # for batch_size in batch_size_list:
            #     in_batch_index += batch_size
            #     sample_image_list.append(image[in_batch_index - sample_num : in_batch_index].flatten())
            #     sample_in_list.append(in_text[in_batch_index - sample_num : in_batch_index].flatten())
            #     sample_out_list.append(out_text[in_batch_index - sample_num : in_batch_index].flatten())
            print_batch_size = f"batch_size = image:{image.shape} || in_text:{in_text.shape} || out_text:{out_text.shape}"
            logger.info(f"{prefix_text}{print_batch_size}")
            print_sample = f"image:{sample_image_list} || in_text:{sample_in_list} || out_text:{sample_out_list}"
            logger.info(f"{prefix_text}{print_sample}")

            # if step == min_step:
            #     logger.info(f"full_image:{image} || full_in_text:{in_text} || full_out_text:{out_text}")

            # if i % args.accumulation_steps == 0:
            #     optimizer.zero_grad()
            # src_images = src_images.to(device_id, non_blocking=True)
            # # if args.pretrain:
            # #     tgt_images = tgt_images.to(device_id)
            # #     tgt_texts, _ = model.module.image_to_z(tgt_images)
            # src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(
            #     device_id, non_blocking=True
            # )  # ['pt', 'tf', 'np', 'jax']
            # tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(
            #     device_id, non_blocking=True
            # )  # ['pt', 'tf', 'np', 'jax']

            out = model(image, in_text)
            loss = criterion(out, out_text)
            loss.backward()
            optimizer.step()

    #         loss /= args.accumulation_steps
    #         loss.backward()

    #         train_loss += loss.item() * src_images.shape[0]
    #         train_count += src_images.shape[0]

    #         # args.accumulation_steps回の勾配を蓄積してから、optimizer.step()を呼び出す
    #         if (i + 1) % args.accumulation_steps == 0 or i + 1 == len(train_loader):
    #             optimizer.step()
    #             pbar.update(1)
    #             if rank == 0:
    #                 steps += 1
    #             if args.num_epochs is None:
    #                 scheduler.step()

    #     # 他のノードから集める
    #     dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(train_count, op=dist.ReduceOp.SUM)

    #     if rank == 0:
    #         train_loss /= train_count
    #         loss_counter.add("train", train_loss.cpu().numpy().copy())

    #     if args.lr_scheduler != '' and args.num_steps is None:
    #         scheduler.step()
    #     pbar.close()
    #     # 検証ループ
    #     if args.image_model_train:
    #         model.module.image_model.eval()
    #     model.module.transformer.eval()
    #     val_loss = torch.tensor(0.0).to(device_id)
    #     val_count = torch.tensor(0).to(device_id)
    #     val_loop = tqdm(val_loader, desc=f'Val (Epoch {epoch}/{args.num_epochs})', disable=(rank != 0))
    #     for src_images, tgt_images, src_texts, tgt_texts in val_loop:
    #         with torch.no_grad():
    #             src_images = src_images.to(device_id, non_blocking=True)
    #             # if args.pretrain:
    #             #    tgt_images = tgt_images.to(device_id)
    #             #    tgt_texts, _ = model.module.image_to_z(tgt_images)
    #             src_texts = src_tokenizer(src_texts, padding="longest", max_length=args.max_source_length, return_tensors='pt')['input_ids'].to(
    #                 device_id, non_blocking=True
    #             )  # ['pt', 'tf', 'np', 'jax']
    #             tgt_texts = tgt_tokenizer(tgt_texts, padding="longest", max_length=args.max_target_length, return_tensors='pt')['input_ids'].to(
    #                 device_id, non_blocking=True
    #             )  # ['pt', 'tf', 'np', 'jax']

    #             loss = model(src_images, src_texts, tgt_texts)

    #             val_loss += loss.item() * src_images.shape[0]
    #             val_count += src_images.shape[0]

    #     # 他のノードから集める
    #     dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    #     dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

    #     if rank == 0:
    #         val_loss /= val_count
    #         loss_counter.add("val", val_loss.cpu().numpy().copy())
    #         logger.info(
    #             f'[Epoch ({epoch}/{args.num_epochs})] Train loss : {train_loss}, Val loss : {val_loss}, Steps : {steps}, Image Mask Ratio : {image_mask_ratio}'
    #         )

    #         if val_loss < min_val_loss:
    #             min_val_loss = val_loss
    #             print('Best Model saving...')
    #             model.module.save()
    #             logger.info('Best Model saved')

    #         if args.save_interval is not None:
    #             if args.num_steps is None:
    #                 if (epoch) % args.save_interval == 0:
    #                     print(f'Model {epoch} saving...')
    #                     model.module.save(result_name=f'epoch_{epoch}.pth')
    #                     print(f'Model {epoch} saved')
    #             else:
    #                 if steps % args.save_interval == 0:
    #                     print(f'Model {steps} saving...')
    #                     model.module.save(result_name=f'step_{steps}.pth')
    #                     print(f'Model {steps} saved')

    # if rank == 0:
    #     loss_counter.plot_loss(args.result_dir)


if __name__ == "__main__":
    # master_addr    = os.getenv('MASTER_ADDR', default='localhost')
    # master_port    = os.getenv('MASTER_PORT', default=27890)
    # dist_url       = 'tcp://{}:{}'.format(master_addr, master_port)
    port_num = 50000
    host_list_file = os.environ["PJM_O_NODEINF"]
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])#dist.get_rank()
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"]) 
    mp.spawn(train, nprocs=4, args=(world_rank,local_rank, world_size,port_num,host_list_file))
