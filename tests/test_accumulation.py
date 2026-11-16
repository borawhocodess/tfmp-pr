import pytest
import schedulefree
import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM


class TablePrior(Prior):
    """
    gives fixed random tables in order, so batches of 1 and 2 hold the same tables
    """

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(8, 30, 4, generator=generator)
        self.y = torch.randint(0, 3, (8, 30), generator=generator).float()
        self.next = 0

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[self.next : self.next + batch_size]
        y = self.y[self.next : self.next + batch_size]
        self.next += batch_size
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def train(batch_size: int, accumulation_steps: int, experiments_dir) -> NanoTabPFNModel:
    """
    trains small classifier for 2 steps on the same 4 tables
    """
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=3
        )
    )
    return pretrainTFM(
        "classification",
        model,
        TablePrior(),
        training=ClassificationTrainingConfig(
            batch_size=batch_size, steps=2, eval_every=2, grad_clip=1e9, accumulation_steps=accumulation_steps
        ),
        experiment=ClassificationExperimentConfig(experiments_dir=str(experiments_dir)),
        device="cpu",
    )


def test_accumulated_batches_match_one_larger_batch(tmp_path, monkeypatch):
    """Two batches of 1 table with accumulation should give the gradients, logged loss and weights of one batch of 2
    tables."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    gradients = []
    step = schedulefree.AdamWScheduleFree.step

    def recording_step(self, *args):
        gradients.append([p.grad.clone() for group in self.param_groups for p in group["params"]])
        return step(self, *args)

    monkeypatch.setattr(schedulefree.AdamWScheduleFree, "step", recording_step)
    losses = []
    on_eval = ClassifierExperimentEvaluationCallback.on_eval

    def recording_on_eval(self, step, train_time, loss, model, **kwargs):
        losses.append(loss)
        return on_eval(self, step, train_time, loss, model, **kwargs)

    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "on_eval", recording_on_eval)
    larger = train(2, 1, tmp_path).state_dict()
    accumulated = train(1, 2, tmp_path).state_dict()

    assert len(gradients) == 4
    assert len(losses) == 2
    assert abs(losses[1] - losses[0]) < 1e-5
    for a, b in zip(gradients[:2], gradients[2:], strict=True):
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(y, x)
    for name, value in larger.items():
        torch.testing.assert_close(accumulated[name], value, rtol=1e-4, atol=1e-4)


def test_accumulation_steps_below_one_are_rejected(tmp_path):
    """Fewer than 1 accumulated batch per step cannot train, so training should stop before it starts."""
    with pytest.raises(ValueError, match="accumulation_steps"):
        train(1, 0, tmp_path)
