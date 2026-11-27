import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM


class NaNEveryOtherPrior(Prior):
    """
    gives random tables, every other one (starting with the first) with a nan feature
    """

    def __init__(self) -> None:
        self.calls = 0

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls += 1
        x = torch.randn(batch_size, 30, 4)
        if self.calls % 2 == 1:
            x[0, 0, 0] = torch.nan
        y = torch.randint(0, 3, (batch_size, 30)).float()
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def test_skipped_batches_are_logged(tmp_path, monkeypatch):
    """A run whose prior gives a nan table every other batch logs each skipped batch."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
        )
    )
    pretrainTFM(
        "classification",
        model,
        NaNEveryOtherPrior(),
        training=ClassificationTrainingConfig(steps=4, eval_every=2),
        experiment=ClassificationExperimentConfig(name="skipped", experiments_dir=str(tmp_path)),
        device="cpu",
    )
    (log,) = (tmp_path / "classification" / "skipped").glob("*/*-log.txt")
    assert [line.rsplit("skipped ", 1)[1] for line in log.read_text().splitlines() if line.startswith("step")] == [
        "2",
        "2",
    ]
