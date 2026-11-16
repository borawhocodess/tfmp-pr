import numpy as np
import pytest
import torch

from tfmplayground.priors.base import Prior, PriorDataLoader, seed_worker


class NumpyPrior(Prior):
    """
    gives random tables from global numpy randomness
    """

    device = "cpu"

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.random.standard_normal((batch_size, 6, 2)))
        y = torch.from_numpy(np.random.standard_normal((batch_size, 6)))
        return x[:, :4], y[:, :4], x[:, 4:], y[:, 4:]


class StatefulPrior(NumpyPrior):
    """
    keeps position, like a prior that reads a dump
    """

    def state_dict(self) -> dict:
        return {"pointer": 0}


def test_workers_make_different_tables():
    """Workers should not repeat each other's tables."""
    batches = iter(PriorDataLoader(NumpyPrior(), batch_size=1, num_workers=2))
    features = [next(batches)[0].numpy().tobytes() for _ in range(4)]

    assert len(set(features)) == 4


def test_worker_seed_follows_torch_seed():
    """A worker's numpy tables should follow its torch seed, so a seeded run gives the same tables again."""
    tables = []
    for seed in (0, 0, 1):
        torch.manual_seed(seed)
        seed_worker(0)
        tables.append(NumpyPrior().batch(1)[0])

    torch.testing.assert_close(tables[0], tables[1], rtol=0, atol=0)
    assert not torch.equal(tables[0], tables[2])


def test_workers_reject_prior_with_state():
    """A prior that keeps state cannot be split over workers, since each worker would keep its own copy."""
    with pytest.raises(ValueError, match="keeps state"):
        PriorDataLoader(StatefulPrior(), batch_size=1, num_workers=2)
