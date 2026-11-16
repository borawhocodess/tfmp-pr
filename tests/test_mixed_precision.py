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


@pytest.mark.parametrize(("mixed_precision", "dtype"), [(True, torch.bfloat16), (False, torch.float32)])
def test_mixed_precision_runs_forward_in_bfloat16(tmp_path, monkeypatch, mixed_precision, dtype):
    """Training forwards should run in bfloat16 only when asked, and weights should stay finite float32."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16,
            num_attention_heads=2,
            mlp_hidden_size=32,
            num_layers=2,
            num_outputs=3,
        )
    )
    forward_dtypes = set()
    model.register_forward_hook(lambda module, args, output: forward_dtypes.add(output.dtype))

    pretrainTFM(
        "classification",
        model,
        RandomPrior(),
        training=ClassificationTrainingConfig(steps=2, eval_every=2, mixed_precision=mixed_precision),
        experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
        device="cpu",
    )

    assert forward_dtypes == {dtype}
    for weights in model.state_dict().values():
        assert weights.dtype == torch.float32
        assert torch.isfinite(weights).all()
