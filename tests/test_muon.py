import pytest
import torch

from tfmplayground import models
from tfmplayground.configs import models as model_configs
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


@pytest.mark.parametrize("name", ["NanoTabPFN", "ModdedNanoTabPFN", "NanoTabICL", "TabICL", "TabFM"])
def test_io_modules_leave_hidden_matrices_to_muon(name):
    """Every input and output module should exist, and muon should get weight matrices only outside of them."""
    model = getattr(models, f"{name}Model")(config=getattr(model_configs, f"{name}ClassifierConfig")())
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    hidden = {names[id(parameter)] for parameter in model.hidden_matrices()}

    assert model.io_modules
    for module in model.io_modules:
        assert any(name.startswith(f"{module}.") for name in names.values()), module
        assert not any(name.startswith(f"{module}.") for name in hidden), module
    assert hidden


def train(optimizer: str, experiments_dir) -> NanoTabPFNModel:
    """
    trains small classifier for 2 steps with optimizer
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
        training=ClassificationTrainingConfig(steps=2, eval_every=2, optimizer=optimizer, weight_decay=0.01),
        experiment=ClassificationExperimentConfig(experiments_dir=str(experiments_dir)),
        device="cpu",
    )


def test_muon_trains_every_parameter(tmp_path, monkeypatch):
    """Training with muon should change hidden matrices and every other parameter, and keep them finite."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    torch.manual_seed(0)
    initial = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16,
            num_attention_heads=2,
            mlp_hidden_size=32,
            num_layers=2,
            num_outputs=3,
        )
    ).state_dict()
    trained = train("muon", tmp_path).state_dict()

    for name, value in trained.items():
        assert torch.isfinite(value).all(), name
        if value.dtype.is_floating_point and name.endswith(("weight", "bias")):
            assert not torch.equal(value, initial[name]), name


def test_unknown_optimizer_is_rejected(tmp_path):
    """An optimizer other than schedulefree_adamw or muon should stop training before it starts."""
    with pytest.raises(ValueError, match="optimizer"):
        train("sgd", tmp_path)
