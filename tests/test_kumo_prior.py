import pytest
import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.priors import KumoClassificationPriorConfig, KumoRegressionPriorConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import KumoPrior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM


@pytest.mark.parametrize(
    "config",
    [
        KumoClassificationPriorConfig(min_num_datapoints=100, max_num_datapoints=150, max_num_features=8),
        KumoRegressionPriorConfig(min_num_datapoints=100, max_num_datapoints=150, max_num_features=8),
    ],
)
def test_batches_are_finite_and_split(config):
    """Batches should share their sizes, stay finite and split every table at the same train test index."""
    torch.manual_seed(0)
    prior = KumoPrior(config=config, device="cpu")
    for _ in range(3):
        x_train, y_train, x_test, y_test = prior.batch(2)

        assert x_train.shape[:2] == y_train.shape and x_test.shape[:2] == y_test.shape
        assert config.min_num_features <= x_train.shape[2] == x_test.shape[2] <= config.max_num_features
        assert config.min_num_datapoints <= x_train.shape[1] + x_test.shape[1] <= config.max_num_datapoints
        assert all(torch.isfinite(t).all() for t in (x_train, y_train, x_test, y_test))


def test_every_test_class_is_in_the_train_part():
    """Classification tables should hold every test class in the train part and use at most max_num_classes classes."""
    torch.manual_seed(0)
    config = KumoClassificationPriorConfig(
        min_num_datapoints=100,
        max_num_datapoints=100,
        max_num_features=6,
        min_train_fraction=0.1,
        max_train_fraction=0.15,
        max_num_classes=10,
    )
    prior = KumoPrior(config=config, device="cpu")
    for _ in range(10):
        _, y_train, _, y_test = prior.batch(2)
        for train, test in zip(y_train, y_test, strict=True):
            assert set(test.tolist()) <= set(train.tolist())
            assert len(train.unique()) >= 2 and train.max() < config.max_num_classes


def test_row_count_is_log_uniform():
    """Row counts should be log-uniform like the kumo training loader, so their median is near the geometric mean."""
    torch.manual_seed(0)
    config = KumoClassificationPriorConfig(min_num_datapoints=400, max_num_datapoints=10240)
    prior = KumoPrior(config=config, device="cpu")
    counts = []
    for _ in range(2000):
        prior.batch_hyperparameters()
        counts.append(prior.num_datapoints)

    assert min(counts) >= 400 and max(counts) <= 10240
    assert 1600 < torch.tensor(counts).float().median() < 2500


@pytest.mark.parametrize(
    "config",
    [
        KumoClassificationPriorConfig(min_train_fraction=0.0),
        KumoClassificationPriorConfig(min_num_features=0),
        KumoClassificationPriorConfig(min_num_datapoints=1),
        KumoClassificationPriorConfig(max_num_classes=1),
    ],
)
def test_impossible_limits_are_rejected(config):
    """Limits that cannot give a valid table should stop before sampling."""
    with pytest.raises(ValueError):
        KumoPrior(config=config, device="cpu")


def test_prior_trains_a_model(tmp_path, monkeypatch):
    """A small model should train on the prior through pretrainTFM."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10
        )
    )
    config = KumoClassificationPriorConfig(min_num_datapoints=100, max_num_datapoints=100, max_num_features=6)
    pretrainTFM(
        "classification",
        model,
        KumoPrior(config=config, device="cpu"),
        training=ClassificationTrainingConfig(steps=2),
        experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
        device="cpu",
    )


@pytest.mark.parametrize("problem_config", [KumoClassificationPriorConfig, KumoRegressionPriorConfig])
def test_missing_values_become_mean_and_indicators(problem_config):
    """With missing_values, batches stay finite and some tables get indicator columns beyond the sampled features."""
    torch.manual_seed(0)
    config = problem_config(min_num_datapoints=100, max_num_datapoints=150, max_num_features=8, missing_values=True)
    prior = KumoPrior(config=config, device="cpu")
    widened = 0
    for _ in range(10):
        x_train, y_train, x_test, y_test = prior.batch(2)
        assert all(torch.isfinite(t).all() for t in (x_train, y_train, x_test, y_test))
        assert x_train.shape[2] <= 2 * prior.num_features
        widened += x_train.shape[2] > prior.num_features
    assert widened > 0
