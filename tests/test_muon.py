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


TINY_CONFIGS = {
    "NanoTabPFN": dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2),
    "ModdedNanoTabPFN": dict(l=2, a=2, e=16, h=32, thinking_rows=4),
    "NanoTabICL": dict(
        embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1, col_nhead=2, row_nhead=2, icl_nhead=2
    ),
    "TabICL": dict(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
    ),
    "TabFM": dict(),
}


@pytest.mark.parametrize("name", TINY_CONFIGS)
def test_io_modules_leave_hidden_matrices_to_muon(name):
    """Every input and output module should exist, and muon should get weight matrices only outside of them."""
    config = getattr(model_configs, f"{name}ClassifierConfig")(**TINY_CONFIGS[name])
    model = getattr(models, f"{name}Model")(config=config)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    hidden = {names[id(parameter)] for parameter in model.hidden_matrices()}

    assert model.io_modules
    for module in model.io_modules:
        assert any(name.startswith(f"{module}.") for name in names.values()), module
        assert not any(name.startswith(f"{module}.") for name in hidden), module
    assert hidden


def train(experiments_dir) -> NanoTabPFNModel:
    """
    trains small classifier for 2 steps with muon
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
        training=ClassificationTrainingConfig(steps=2, eval_every=2, muon=True, weight_decay=0.01),
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
    trained = train(tmp_path).state_dict()

    for name, value in trained.items():
        assert torch.isfinite(value).all(), name
        if value.dtype.is_floating_point and name.endswith(("weight", "bias")):
            assert not torch.equal(value, initial[name]), name
