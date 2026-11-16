import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training import pretrain
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback


class RandomPrior(Prior):
    """
    gives random tables from global torch randomness
    """

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, 30, 4)
        y = torch.randint(0, 3, (batch_size, 30)).float()
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def train(grad_checkpointing: bool, experiments_dir) -> NanoTabPFNModel:
    """
    trains small classifier with 2 blocks for 2 steps
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
    return pretrain.pretrainTFM(
        "classification",
        model,
        RandomPrior(),
        training=ClassificationTrainingConfig(steps=2, eval_every=2, grad_checkpointing=grad_checkpointing),
        experiment=ClassificationExperimentConfig(experiments_dir=str(experiments_dir)),
        device="cpu",
    )


def test_checkpointed_blocks_train_to_same_weights(tmp_path, monkeypatch):
    """Training with gradient checkpointing should checkpoint every block in every step and give the same weights."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    calls = []
    checkpoint = pretrain.checkpoint

    def counted(*args, **kwargs):
        calls.append(1)
        return checkpoint(*args, **kwargs)

    monkeypatch.setattr(pretrain, "checkpoint", counted)

    plain = train(False, tmp_path)
    assert not calls
    checkpointed = train(True, tmp_path)

    assert len(calls) == 2 * 2
    assert not any("forward" in vars(block) for block in checkpointed.transformer_blocks)
    for name, weights in plain.state_dict().items():
        torch.testing.assert_close(checkpointed.state_dict()[name], weights, rtol=0, atol=0)
