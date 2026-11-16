import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM


class CountingPrior(Prior):
    """
    gives table number i from seed i, so batch size does not change which tables come
    """

    def __init__(self) -> None:
        self.tables = 0

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(batch_size):
            generator = torch.Generator().manual_seed(self.tables)
            xs.append(torch.randn(30, 4, generator=generator))
            ys.append(torch.randint(0, 3, (30,), generator=generator).float())
            self.tables += 1
        x, y = torch.stack(xs), torch.stack(ys)
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def train(batch_size: int, grad_accumulation: int, experiments_dir) -> tuple[NanoTabPFNModel, int, str]:
    """
    trains small classifier for 2 steps, gives it back with number of tables it saw and loss it logged
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
    prior = CountingPrior()
    training = ClassificationTrainingConfig(
        steps=2, eval_every=2, batch_size=batch_size, grad_accumulation=grad_accumulation
    )
    experiment = ClassificationExperimentConfig(experiments_dir=str(experiments_dir))
    model = pretrainTFM("classification", model, prior, training=training, experiment=experiment, device="cpu")
    (log,) = experiments_dir.glob("**/*-log.txt")
    return model, prior.tables, log.read_text().split("mean loss ")[1].split(" ")[0]


def distance(a: NanoTabPFNModel, b: NanoTabPFNModel) -> float:
    """
    gives largest difference between weights of two models
    """
    return max((weights - b.state_dict()[name]).abs().max().item() for name, weights in a.state_dict().items())


def test_accumulated_batches_train_like_one_large_batch(tmp_path, monkeypatch):
    """Two accumulated batches of 2 tables should train like one batch of 4 tables, up to float rounding."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    large, large_tables, large_loss = train(batch_size=4, grad_accumulation=1, experiments_dir=tmp_path / "large")
    accumulated, tables, loss = train(batch_size=2, grad_accumulation=2, experiments_dir=tmp_path / "accumulated")
    small, _, _ = train(batch_size=2, grad_accumulation=1, experiments_dir=tmp_path / "small")

    assert tables == large_tables == 8
    assert loss == large_loss
    assert distance(accumulated, large) < 0.1 * distance(small, large)
