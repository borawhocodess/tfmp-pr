import pytest
import torch

from tfmplayground.configs.models import NanoTabPFNClassifierConfig
from tfmplayground.configs.priors import SCMClassificationPriorConfig, SCMRegressionPriorConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.priors import SCMPrior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM


@pytest.mark.parametrize(
    "config",
    [
        SCMClassificationPriorConfig(max_num_datapoints=300, max_num_features=10),
        SCMRegressionPriorConfig(max_num_datapoints=300, max_num_features=10),
    ],
)
def test_batches_are_finite_and_split(config):
    """Batches should share their sizes, stay finite and split every table at the same train test index."""
    torch.manual_seed(0)
    prior = SCMPrior(config=config, device="cpu")
    for _ in range(10):
        x_train, y_train, x_test, y_test = prior.batch(2)

        assert x_train.shape[:2] == y_train.shape and x_test.shape[:2] == y_test.shape
        assert x_train.shape[2] == x_test.shape[2] <= config.max_num_features
        assert config.min_num_datapoints <= x_train.shape[1] + x_test.shape[1] <= config.max_num_datapoints
        assert all(torch.isfinite(t).all() for t in (x_train, y_train, x_test, y_test))


def test_every_test_class_is_in_the_train_part():
    """Classification tables should hold every class in the train part and use at most max_num_classes classes."""
    torch.manual_seed(0)
    config = SCMClassificationPriorConfig(
        min_num_datapoints=40,
        max_num_datapoints=60,
        max_num_features=10,
        min_train_fraction=0.1,
        max_train_fraction=0.15,
        max_num_classes=4,
    )
    prior = SCMPrior(config=config, device="cpu")
    for _ in range(50):
        _, y_train, _, y_test = prior.batch(2)
        for train, test in zip(y_train, y_test, strict=True):
            assert set(test.tolist()) <= set(train.tolist())
            assert train.max() < config.max_num_classes


@pytest.mark.parametrize(
    "config",
    [
        SCMClassificationPriorConfig(min_train_fraction=0.0),
        SCMClassificationPriorConfig(min_num_features=0),
        SCMClassificationPriorConfig(min_num_datapoints=20),
        SCMRegressionPriorConfig(min_num_datapoints=10),
    ],
)
def test_impossible_limits_are_rejected(config):
    """Limits that cannot give a train part with enough rows should stop before sampling."""
    with pytest.raises(ValueError):
        SCMPrior(config=config, device="cpu")


def test_prior_trains_a_model(tmp_path, monkeypatch):
    """A small model should train on the prior through pretrainTFM."""
    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", lambda self, model, **kwargs: [0.5])
    torch.manual_seed(0)
    model = NanoTabPFNModel(
        config=NanoTabPFNClassifierConfig(
            embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10
        )
    )
    pretrainTFM(
        "classification",
        model,
        SCMPrior(config=SCMClassificationPriorConfig(max_num_datapoints=200, max_num_features=8), device="cpu"),
        training=ClassificationTrainingConfig(steps=2),
        experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
        device="cpu",
    )
