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


def test_warmup_rises_linearly_before_cosine():
    """Warmup should rise from 0 to lr over its steps, then cosine should decay over the remaining steps."""
    training = ClassificationTrainingConfig(lr=1e-3, steps=10, lr_schedule="cosine", warmup_proportion=0.2)

    assert learning_rate(training, 0) == 0.0
    assert math.isclose(learning_rate(training, 1), 5e-4)
    assert learning_rate(training, 2) == 1e-3
    assert math.isclose(learning_rate(training, 6), 5e-4)
    assert math.isclose(learning_rate(training, 10), 0.0, abs_tol=1e-12)


def test_wsd_stays_constant_then_decays_linearly():
    """Wsd should rise over the warmup, stay at lr, then fall linearly to 0 over the last decay_proportion of steps."""
    training = ClassificationTrainingConfig(
        lr=1e-3, steps=10, lr_schedule="wsd", warmup_proportion=0.2, decay_proportion=0.4
    )

    assert math.isclose(learning_rate(training, 1), 5e-4)
    assert learning_rate(training, 2) == 1e-3
    assert learning_rate(training, 6) == 1e-3
    assert math.isclose(learning_rate(training, 8), 5e-4)
    assert learning_rate(training, 10) == 0.0


@pytest.mark.parametrize("muon", [False, True])
def test_training_sets_scheduled_learning_rate(tmp_path, monkeypatch, muon):
    """Every optimizer step should use the learning rate of the schedule at that step."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    seen = []
    monkeypatch.setattr(
        "schedulefree.AdamWScheduleFree.step", lambda self, *args: seen.append(self.param_groups[0]["lr"])
    )
    training = ClassificationTrainingConfig(
        lr=1e-3, steps=4, eval_every=2, muon=muon, lr_schedule="cosine", warmup_proportion=0.5
    )
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


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"lr_schedule": "linear"}, "schedule"),
        ({"warmup_proportion": 1.0}, "warmup"),
        ({"lr_schedule": "wsd"}, "decay"),
        ({"lr_schedule": "wsd", "warmup_proportion": 0.5, "decay_proportion": 0.6}, "decay"),
    ],
)
def test_unknown_schedule_is_rejected(tmp_path, settings, message):
    """
    A schedule other than constant, cosine or wsd, a warmup of all steps, or a wsd decay that is empty or overlaps the
    warmup should stop training before it starts.
    """
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
        )
    )
    with pytest.raises(ValueError, match=message):
        pretrainTFM(
            "classification",
            model,
            RandomPrior(),
            training=ClassificationTrainingConfig(steps=2, **settings),
            experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
            device="cpu",
        )
