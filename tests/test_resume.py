import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training import pretrainTFM
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback


class FixedPrior(Prior):
    """
    gives same table every time, so runs do not depend on random state
    """

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(1, 12, 3, generator=generator)
        self.y = torch.randint(0, 2, (1, 12), generator=generator).float()

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[:, :8], self.y[:, :8], self.x[:, 8:], self.y[:, 8:]


def run(tmp_path, steps: int, resume: str | None = None) -> tuple[NanoTabPFNModel, list]:
    """
    pretrains tiny classifier for steps, gives it back with checkpoints of its run
    """
    torch.manual_seed(0)
    config = NanoTabPFNClassifierConfig(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=2
    )
    model = pretrainTFM(
        problem="classification",
        model=NanoTabPFNModel(config=config),
        prior=FixedPrior(),
        training=ClassificationTrainingConfig(steps=steps, eval_every=2),
        experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path), resume=resume),
        device="cpu",
    )
    return model, sorted(tmp_path.glob("**/*-ckpt-last.pth"))


def test_resumed_run_matches_uninterrupted_run(tmp_path, monkeypatch):
    """A run stopped at step 2 and resumed to step 4 should end with the weights of a run that went to step 4."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])

    whole, _ = run(tmp_path / "whole", steps=4)
    _, checkpoints = run(tmp_path / "parts", steps=2)
    resumed, checkpoints_after = run(tmp_path / "parts", steps=4, resume=str(checkpoints[0]))

    assert checkpoints_after == checkpoints
    assert torch.load(checkpoints[0])["step"] == 4
    for name, weights in whole.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], weights, rtol=0, atol=0)
