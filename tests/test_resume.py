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


def train(
    steps: int, experiments_dir, resume: str | None = None, optimizer: str = "schedulefree_adamw"
) -> NanoTabPFNModel:
    """
    trains small classifier for steps, from scratch or from resumed run
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
        training=ClassificationTrainingConfig(steps=steps, eval_every=2, optimizer=optimizer),
        experiment=ClassificationExperimentConfig(name="resume", experiments_dir=str(experiments_dir), resume=resume),
        device="cpu",
    )


@pytest.mark.parametrize("optimizer", ["schedulefree_adamw", "muon"])
def test_resumed_training_matches_uninterrupted_training(tmp_path, monkeypatch, optimizer):
    """Training 2 steps and resuming for 2 more should give the same model as training 4 steps at once."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    uninterrupted = train(4, tmp_path / "uninterrupted", optimizer=optimizer)
    train(2, tmp_path / "interrupted", optimizer=optimizer)
    (run_dir,) = (tmp_path / "interrupted" / "classification" / "resume").iterdir()
    resumed = train(4, tmp_path / "interrupted", resume=str(run_dir), optimizer=optimizer)

    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
