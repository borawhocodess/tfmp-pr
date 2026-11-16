import random
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class Prior(ABC):
    """
    base class for every prior this package trains on
    """

    config = None

    @abstractmethod
    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gives one batch of tables, split into train and test parts
        """
        ...

    def state_dict(self) -> dict:
        """
        gives what prior needs to continue where it stopped, beyond global randomness
        """
        return {}

    def load_state_dict(self, state: dict) -> None:
        """
        continues prior from what state_dict gave
        """
        if state:
            raise ValueError(f"{type(self).__name__} keeps no state, not {sorted(state)}")


class CPUPriorBatches(IterableDataset):
    """
    gives batches from prior without end on cpu, so worker processes can make them
    """

    def __init__(self, prior: Prior, batch_size: int) -> None:
        """
        keeps prior and batch size
        """
        self.prior = prior
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        moves this worker's copy of prior to cpu, then gives its batches
        """
        self.prior.device = "cpu"
        while True:
            yield self.prior.batch(self.batch_size)


def seed_worker(worker_id: int) -> None:
    """
    seeds python and numpy randomness of worker from its torch seed, so workers make different tables
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class PriorDataLoader:
    """
    turns prior into endless iterator of batches, made in worker processes when there are workers
    """

    def __init__(self, prior: Prior, batch_size: int, num_workers: int = 0) -> None:
        """
        keeps prior, batch size and number of workers, and checks that workers can make this prior's batches
        """
        if num_workers < 0:
            raise ValueError(f"number of workers must be at least 0, not {num_workers}")
        if num_workers > 0 and prior.state_dict():
            raise ValueError(f"{type(prior).__name__} keeps state, so worker processes cannot make its batches")
        self.prior = prior
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        gives batches from prior without end
        """
        if self.num_workers == 0:
            while True:
                yield self.prior.batch(self.batch_size)
        yield from DataLoader(
            CPUPriorBatches(self.prior, self.batch_size),
            batch_size=None,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
        )
