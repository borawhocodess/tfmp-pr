import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import default_optimizer, pretrainTFM


class RandomPrior(Prior):
    """
    gives random tables from global torch randomness
    """

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, 30, 4)
        y = torch.randint(0, 3, (batch_size, 30)).float()
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def small_model() -> NanoTabPFNModel:
    """
    gives small classifier with same weights every time
    """
    torch.manual_seed(0)
    return NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16,
            num_attention_heads=2,
            mlp_hidden_size=32,
            num_layers=2,
            num_outputs=3,
        )
    )


def train(steps: int, experiments_dir, resume: str | None = None) -> NanoTabPFNModel:
    """
    trains small classifier with muon for steps, from scratch or from resumed run
    """
    return pretrainTFM(
        "classification",
        small_model(),
        RandomPrior(),
        training=ClassificationTrainingConfig(steps=steps, eval_every=2, muon=True),
        experiment=ClassificationExperimentConfig(name="muon", experiments_dir=str(experiments_dir), resume=resume),
        device="cpu",
    )


def test_muon_takes_2d_weights_and_adamw_takes_the_rest():
    """Every weight should be in exactly one optimizer, 2d weights in muon."""
    model = small_model()
    muon, adamw = default_optimizer(model, ClassificationTrainingConfig(muon=True)).optimizers

    in_muon = [p for group in muon.param_groups for p in group["params"]]
    in_adamw = [p for group in adamw.param_groups for p in group["params"]]

    assert isinstance(muon, torch.optim.Muon)
    assert all(p.ndim == 2 for p in in_muon)
    assert all(p.ndim != 2 for p in in_adamw)
    assert len(in_muon) + len(in_adamw) == len(list(model.parameters()))


def test_resumed_muon_training_matches_uninterrupted_training(tmp_path, monkeypatch):
    """Training 2 steps with muon and resuming for 2 more should give the same model as training 4 steps at once."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    start = small_model().state_dict()
    uninterrupted = train(4, tmp_path / "uninterrupted")
    train(2, tmp_path / "interrupted")
    (run_dir,) = (tmp_path / "interrupted" / "classification" / "muon").iterdir()
    resumed = train(4, tmp_path / "interrupted", resume=str(run_dir))

    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    assert any(not torch.equal(value, start[name]) for name, value in uninterrupted.state_dict().items())
