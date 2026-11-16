import pytest
import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM


class RandomPrior(Prior):
    """
    gives random tables from global torch randomness
    """

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, 30, 4)
        y = torch.randint(0, 3, (batch_size, 30)).float()
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def train(precision: str, experiments_dir) -> NanoTabPFNModel:
    """
    trains small classifier for 2 steps in precision
    """
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16,
            num_attention_heads=2,
            mlp_hidden_size=32,
            num_layers=2,
            num_outputs=3,
        )
    )
    return pretrainTFM(
        "classification",
        model,
        RandomPrior(),
        training=ClassificationTrainingConfig(steps=2, eval_every=2, precision=precision),
        experiment=ClassificationExperimentConfig(experiments_dir=str(experiments_dir)),
        device="cpu",
    )


def test_bfloat16_training_keeps_float32_weights(tmp_path, monkeypatch):
    """Training in bfloat16 should keep float32 weights and give other weights than training in float32."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    full = train("float32", tmp_path).state_dict()
    mixed = train("bfloat16", tmp_path).state_dict()

    assert all(value.dtype == full[name].dtype for name, value in mixed.items())
    assert any(not torch.equal(value, full[name]) for name, value in mixed.items())


def test_unknown_precision_is_rejected(tmp_path):
    """A precision other than float32 or bfloat16 should stop training before it starts."""
    with pytest.raises(ValueError, match="precision"):
        train("float16", tmp_path)
