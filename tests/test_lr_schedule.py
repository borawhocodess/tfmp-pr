import math

import pytest
import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import learning_rate, pretrainTFM


class RandomPrior(Prior):
    """
    gives random tables from global torch randomness
    """

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, 30, 4)
        y = torch.randint(0, 3, (batch_size, 30)).float()
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def test_cosine_decays_from_lr_to_zero():
    """Cosine should start at lr, be at half of lr halfway, and reach 0 after the last step."""
    training = ClassificationTrainingConfig(lr=1e-3, steps=10, lr_schedule="cosine")

    assert learning_rate(training, 0) == 1e-3
    assert math.isclose(learning_rate(training, 5), 5e-4)
    assert math.isclose(learning_rate(training, 10), 0.0, abs_tol=1e-12)
    assert learning_rate(ClassificationTrainingConfig(lr=1e-3, steps=10), 7) == 1e-3


@pytest.mark.parametrize("optimizer", ["schedulefree_adamw", "muon"])
def test_training_sets_scheduled_learning_rate(tmp_path, monkeypatch, optimizer):
    """Every optimizer step should use the learning rate of the schedule at that step."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    seen = []
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *args: seen.append(self.param_groups[0]["lr"]))
    monkeypatch.setattr(
        "schedulefree.AdamWScheduleFree.step", lambda self, *args: seen.append(self.param_groups[0]["lr"])
    )
    training = ClassificationTrainingConfig(lr=1e-3, steps=4, eval_every=2, optimizer=optimizer, lr_schedule="cosine")
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
        )
    )
    pretrainTFM(
        "classification",
        model,
        RandomPrior(),
        training=training,
        experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
        device="cpu",
    )

    assert seen == [learning_rate(training, step) for step in range(4)]


def test_unknown_schedule_is_rejected(tmp_path):
    """A schedule other than constant or cosine should stop training before it starts."""
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
        )
    )
    with pytest.raises(ValueError, match="schedule"):
        pretrainTFM(
            "classification",
            model,
            RandomPrior(),
            training=ClassificationTrainingConfig(steps=2, lr_schedule="linear"),
            experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
            device="cpu",
        )
