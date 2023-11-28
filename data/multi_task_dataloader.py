import itertools
import math
import random
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .get_loader import get_dataset
from multiprocessing import Pool

class MultiTaskDataIterator1:
    def __init__(self, dataloader_list, min_step) -> None:
        self.iter_list = [iter(dataloader) for dataloader in dataloader_list]
        self.min_step = min(min_step)
        self.step = 0

    def __next__(self):
        if self.step == self.min_step:
            raise StopIteration

        next_sample = [next(iter) for iter in self.iter_list]  # [taskA,taskB,taskC]

        self.step += 1
        return next_sample

    def __len__(self):
        return self.min_step


class MultiTaskDataLoader1:
    def __init__(
        self,
        dataset_dict: dict[str, DataLoader],
        batch_size_dict: dict[str, int],
        is_ddp=False,
        seed=0,
        loader_drop_last=False,
        sampler_drop_last=False,
        **dataloader_args,
    ) -> None:
        """_summary_

        Args:
            dataset_dict (dict[str,DataLoader]): {taskA:Dataset,taskB:Dataset,taskC:Dataset}のようなdict
            batch_size_dict (_type_): ｛taskA:10,taskB:20,taskC:10｝のようなdict
            each_task_collate_fn_dict (dict[str,function], optional): {taskA:collate_fnA,taskB:collate_fnB,task:C,collate_fnC}のようなdict.タスクごとのデーターローダー Defaults to None.
            multi_task_collate_fn (_type_, optional): すべてのタスクからのバッチを統合する関数 Defaults to None.
            is_ddp (bool, optional): DDPか否か Defaults to False.
            seed (int, optional): 乱数シード Defaults to 0.
        """

        dataset_dict_keys = dataset_dict.keys()
        if is_ddp:
            distributed_keys = ["num_replicas", "rank", "shuffle"]
            distributed_args_dict = {}
            for key in distributed_keys:
                if key in dataloader_args:
                    distributed_args_dict[key] = dataloader_args[key]
                    del dataloader_args[key]

            self.distributed_sampler_dict = {
                key: DistributedSampler(dataset_dict[key], seed=seed, drop_last=sampler_drop_last, **distributed_args_dict) for key in dataset_dict_keys
            }
        else:
            self.distributed_sampler_dict = {key: None for key in dataset_dict_keys}

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        self.dataloader_list = [
            DataLoader(
                dataset_dict[key],
                batch_size_dict[key],
                sampler=self.distributed_sampler_dict[key],
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=loader_drop_last,
                **dataloader_args,
            )
            for key in dataset_dict_keys
        ]
        self.step_list = [len(dataloader) for dataloader in self.dataloader_list]
        self.min_step = min(self.step_list)

    def __iter__(self):
        return MultiTaskDataIterator1(self.dataloader_list, self.step_list)

    def __len__(self):
        return self.min_step

    def set_epoch(self, epoch: int):
        for sampler in self.distributed_sampler_dict.values():
            sampler.set_epoch(epoch)

def get_dataset_proc(item):
    (key, dataset_names), args, phase = item
    datasets = ConcatDataset(
        [
            get_dataset(
                args.root_dir,
                dataset_name,
                args.stage,
                phase=phase,
            )
            for dataset_name in dataset_names
        ]
    )
    return key, datasets

def get_dataset_dict(args, dataset_name_dict: dict[str, List[str]], phase, src_tokenizer=None, tgt_tokenizer=None):
    dataset_dict = {}
    for key in dataset_name_dict.keys():
        dataset_dict[key] = []

    with Pool(8) as p:
        for key, datasets in p.imap_unordered(get_dataset_proc, zip(dataset_name_dict.items(), itertools.repeat(args), itertools.repeat(phase))):
            dataset_dict[key] = datasets
    return dataset_dict


def get_multi_task_data(args, train_dataset_name_dict, phase="train", src_tokenizer=None, tgt_tokenizer=None):
    if len(train_dataset_name_dict) == 0:
        raise ValueError
    dataset_dict = get_dataset_dict(args, train_dataset_name_dict, phase=phase, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    return dataset_dict


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by ``__len``.
            This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self._itr = iter(iterable)
        self.n = start or getattr(iterable, "n", 0)
        self.total = total if total is not None else self.n + len(iterable)

    def __len__(self):
        return self.total

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_next():
            raise StopIteration
        try:
            x = next(self._itr)
        except StopIteration:
            raise IndexError(
                f"Iterator expected to have length {self.total}, "
                f"but exhausted at position {self.n}."
            )
        self.n += 1
        return x

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < self.total

    def skip(self, n):
        """Fast-forward the iterator by skipping n elements."""
        for _ in range(n):
            next(self)
        return self

    def take(self, n):
        """Truncate the iterator to n elements at most."""
        self.total = min(self.total, n)
        # Propagate this change to the underlying iterator
        if hasattr(self._itr, "take"):
            self._itr.take(max(n - self.n, 0))
        return self

class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
        skip_remainder_batch (bool, optional): if set, discard the last grouped batch in
          each training epoch, as the last grouped batch is usually smaller than
                local_batch_size * distributed_word_size * chunk_size (default: ``False``).
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size, skip_remainder_batch=False):
        if skip_remainder_batch:
            total_num_itrs = int(math.floor(len(iterable) / float(chunk_size)))
            #logger.info(
            #    f"skip final residual batch, grouped total_num_itrs = {total_num_itrs}"
            #)
        else:
            #raise NotImplementedError
            total_num_itrs = int(math.ceil(len(iterable) / float(chunk_size)))
            #logger.info(f"grouped total_num_itrs = {total_num_itrs}")

        itr = _chunk_iterator(iterable, chunk_size, skip_remainder_batch)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(chunk_size))),
            total=total_num_itrs,
        )
        self.chunk_size = chunk_size

        if skip_remainder_batch:
            self.take(total_num_itrs)
            # TODO: [Hack] Here the grouped iterator modifies the base iterator size so that
            # training can move into the next epoch once the grouped iterator is exhausted.
            # Double-check this implementation in case unexpected behavior occurs.
            iterable.take(total_num_itrs * chunk_size)


def _chunk_iterator(itr, chunk_size, skip_remainder_batch=False):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if not skip_remainder_batch and len(chunk) > 0:
        yield chunk
        
        
class MultiTaskDataIterator4:
    def __init__(self, dataloader_list, step_list, sample_num_list) -> None:
        self.dataloader_list = dataloader_list
        self.min_step = min(step_list)
        self.step = 0
        self.sample_num_list = sample_num_list
        self.start = 0
        self.grouped_iterator_list = [iter(GroupedIterator(CountingIterator(dataloader,0), sample_num,skip_remainder_batch=True)) for dataloader,sample_num in zip(self.dataloader_list,self.sample_num_list)]

    def __next__(self):
        if self.step == self.min_step:
            raise StopIteration

        next_sample_list = itertools.chain.from_iterable([next(iter) for iter in self.grouped_iterator_list])  # [taskA,taskB,taskC]

        self.step += 1
        return next_sample_list

    def __len__(self):
        return self.min_step


class MultiTaskDataLoader4:
    def __init__(
        self,
        dataset_dict: dict[str, DataLoader],
        batch_size_dict: dict[str, int],
        each_task_sample_num_dict: dict[str, int] = None,
        is_ddp=False,
        seed=0,
        loader_drop_last=False,
        sampler_drop_last=False,
        **dataloader_args,
    ) -> None:
        """_summary_

        Args:
            dataset_dict (dict[str,DataLoader]): {taskA:Dataset,taskB:Dataset,taskC:Dataset}のようなdict
            batch_size_dict (_type_): ｛taskA:10,taskB:20,taskC:10｝のようなdict
            each_task_collate_fn_dict (dict[str,function], optional): {taskA:collate_fnA,taskB:collate_fnB,task:C,collate_fnC}のようなdict.タスクごとのデーターローダー Defaults to None.
            multi_task_collate_fn (_type_, optional): すべてのタスクからのバッチを統合する関数 Defaults to None.
            is_ddp (bool, optional): DDPか否か Defaults to False.
            seed (int, optional): 乱数シード Defaults to 0.
        """

        dataset_dict_keys = dataset_dict.keys()
        if is_ddp:
            distributed_keys = ["num_replicas", "rank", "shuffle"]
            distributed_args_dict = {}
            for key in distributed_keys:
                if key in dataloader_args:
                    distributed_args_dict[key] = dataloader_args[key]
                    del dataloader_args[key]

            self.distributed_sampler_dict = {
                key: DistributedSampler(dataset_dict[key], seed=seed, drop_last=sampler_drop_last, **distributed_args_dict) for key in dataset_dict_keys
            }
        else:
            self.distributed_sampler_dict = {key: None for key in dataset_dict_keys}

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(seed)

        self.dataloader_list = [
            DataLoader(
                dataset_dict[key],
                batch_size_dict[key],
                sampler=self.distributed_sampler_dict[key],
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=loader_drop_last,
                **dataloader_args,
            )
            for key in dataset_dict_keys
        ]
        
        self.sample_num_list = [each_task_sample_num_dict[key] for key in dataset_dict_keys]
        self.step_list = [int(math.floor(len(dataloader) / float(sample_num))) for sample_num,dataloader in zip(self.sample_num_list,self.dataloader_list)]
        self.min_step = min(self.step_list)

    def __iter__(self):
        return MultiTaskDataIterator4(self.dataloader_list, self.step_list, self.sample_num_list)

    def __len__(self):
        return self.min_step

    def set_epoch(self, epoch: int):
        for sampler in self.distributed_sampler_dict.values():
            sampler.set_epoch(epoch)

class DataNumCounter():
    def __init__(self,max_data_num_dict,one_gpu_batch_size_dict,each_task_sample_num_dict,world_size) -> None:
        self.one_gpu_max_data_num_dict = {k: v // world_size for k, v in max_data_num_dict.items()}
        self.one_gpu_data_num_per_step_dict = {k: v * one_gpu_batch_size_dict[k] for k, v in each_task_sample_num_dict.items()}
        self.max_step_dict = {k: v // self.one_gpu_data_num_per_step_dict[k] for k, v in self.one_gpu_max_data_num_dict.items()}
        
        self.accumulate_data_num_dict = {k:0 for k in self.one_gpu_data_num_per_step_dict.keys()}
        self.sample_max_data_flag = False
        self.step = 0
        
    def update(self):
        self.step += 1
        self.accumulate_data_num_dict = {k: v + self.one_gpu_data_num_per_step_dict[k] for k, v in self.accumulate_data_num_dict.items()}
        for task, data_num in self.one_gpu_max_data_num_dict.items():
            if self.accumulate_data_num_dict[task] <= data_num:
                continue
            else:
                self.sample_max_data_flag = True
                
    def reset(self):
        self.accumulate_data_num_dict = {k:0 for k in self.one_gpu_data_num_per_step_dict.keys()}
        self.sample_max_data_flag = False
        self.step = 0
            