# https://github.com/microsoft/unilm/blob/master/kosmos-2/infinibatch/infinibatch/iterators.py#L244

import collections
from abc import abstractmethod


class CheckpointableIterator(collections.abc.Iterator):
    """
    Abstract base class that defines the interface for checkpointing.

    The interface (getstate, setstate) is inspired by Python's random package.
    """

    def __iter__(self) -> 'CheckpointableIterator':
        return self

    @abstractmethod
    def getstate(self) -> dict:
        """
        Get checkpoint of current state of iterator

        In a pipeline of iterators, this function __recursively__ calls itself on the preceeding iterator
        and includes the gathered information in the returned checkpoint.
        Thereby, to obtain a checkpoint of the state of an entire pipeline of iterators
        you only have to call this function on the __last__ iterator in the pipeline.
        A checkpoint is represented as a `dict`,
        but the caller should treat a checkpoint as an opaque object
        and not make any assumptions about the existence or meaning of the `dict` entries.
        """
        pass

    @abstractmethod
    def setstate(self, checkpoint: dict):
        """
        Set state of iterator to given checkpoint

        In a pipeline of iterators, this function __recursively__ calls itself on the preceeding iterator.
        Thereby, to set the state of an entire pipeline of iterators to a given checkpoint
        you only have to call this function on the __last__ iterator in the pipeline.

        Args:
            checkpoint: Checkpoint that should be used to reset the state of the iterator (or pipeline).
                        If this is __None__, the state of the iterator (or pipeline) is reset to the initial
                        state immediately after construction.
        """
        pass

    def __getstate__(self) -> dict:  # implementation of pickle Protocol
        return self.getstate()

    def __setstate__(self, checkpoint: dict):
        self.setstate(checkpoint)

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def close(self):
        """
        Close all PrefetchIterators in this pipeline

        PrefetchIterators have internal resources that need to be properly managed by calling close() manually.
        Failure to do so can lead to dangling processes and threads, or the PrefetchIterator hanging on finalization.
        Note that it is not correct to rely on the garbage collector to destroy PrefetchIterators
        as CPython does not assure that the finalizer (__del__) of a PrefetchIterator will be called.

        This function, which is implemented for every CheckpointableIterator, recursively traverses all preceeding
        iterators and closes all PrefetchIterators in the pipeline.
        For pipelines that do not contain PrefetchIterators this function has no effect.
        """
        pass


# https://github.com/microsoft/unilm/blob/master/kosmos-2/unilm/data/utils.py
from random import Random


class MixIterator(CheckpointableIterator):
    """
    Concat items from all given iterators.
    """

    def __init__(self, source_iterators, weights):
        """
        Args:
                source_iterators: list of iterators to zip, item by item
        """
        # TODO: Use all function?
        for source_iterator in source_iterators:
            if not isinstance(source_iterator, CheckpointableIterator):
                raise ValueError('all iterators in source_iterators have to be CheckpointableIterator')
        self._source_iterators = source_iterators  # type: List[CheckpointableIterator]
        assert len(weights) == len(source_iterators)
        self.weights = weights
        self.population = list(range(len(source_iterators)))

    def getstate(self):
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint):
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            # TODO: Add check that both lists have the same length?
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        _random = Random()
        res = {}  # (note: can't use a generator expression, as it gets confused when a next() call raises StopIteration)
        idx = _random.choices(self.population, self.weights)[0]
        res.update(next(self._source_iterators[idx]))
        return res

    def close(self):
        for it in self._source_iterators:
            it.close()


import numpy as np


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if isinstance(x, np.ndarray):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


class BaseBatchGen(CheckpointableIterator):
    """
    This is a base class for batch generators that use infinibatch
    """

    def __init__(self):
        self._iter = None
        self.epoch = 1
        self.next_epoch_idx = 1
        self.sharded_checkpoint = False
        self.should_close_after_finished = True

    def _build_iter(self):
        """
        Build infinibatch iterator and assign to self._iter
        """
        raise NotImplementedError()

    def _move_to_tensor(self, batch):
        def to_tensor(x):
            return torch.tensor(x)

        return apply_to_sample(to_tensor, batch)

    @property
    def iterator(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __iter__(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __next__(self):
        return next(self._iter)

    def setstate(self, value):
        self._iter.setstate(value)

    def getstate(self):
        return self._iter.getstate()

    def close(self):
        self._iter.close()

    def __len__(self) -> int:
        return 819200000

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True):
        return self

    def end_of_epoch(self) -> bool:
        return False

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return self.getstate()

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.setstate(state_dict)

    @property
    def first_batch(self):
        return "DUMMY"


class MixIterator(CheckpointableIterator):
    """
    Concat items from all given iterators.
    """

    def __init__(self, source_iterators, weights):
        """
        Args:
                source_iterators: list of iterators to zip, item by item
        """
        # TODO: Use all function?
        for source_iterator in source_iterators:
            if not isinstance(source_iterator, CheckpointableIterator):
                raise ValueError('all iterators in source_iterators have to be CheckpointableIterator')
        self._source_iterators = source_iterators  # type: List[CheckpointableIterator]
        assert len(weights) == len(source_iterators)
        self.weights = weights
        self.population = list(range(len(source_iterators)))

    def getstate(self):
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint):
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            # TODO: Add check that both lists have the same length?
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        _random = Random()
        res = {}  # (note: can't use a generator expression, as it gets confused when a next() call raises StopIteration)
        idx = _random.choices(self.population, self.weights)[0]
        res.update(next(self._source_iterators[idx]))
        return res

    def close(self):
        for it in self._source_iterators:
            it.close()


# https://github.com/microsoft/unilm/blob/master/kosmos-2/unilm/data/basic_loader.py#L11
class MixLoader(BaseBatchGen):
    def __init__(self, dataloaders, weights):
        super().__init__()
        self.dataloaders = list(dataloaders)
        self.weights = weights
        self._build_iter()

    def _build_iter(self):
        """
        Build infinibatch iterator and assign to self._iter
        """
        self._iter = MixIterator([dataloader.iterator for dataloader in self.dataloaders], self.weights)
