import os
import logging

def get_logger(args, log_name='train.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')

    # ログのコンソール出力の設定
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # ログのファイル出力先を設定
    log_path = os.path.join(args.result_dir, log_name)
    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Options: {args.__dict__}")

    return logger