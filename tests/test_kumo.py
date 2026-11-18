import numpy as np
import pytest
import torch

from tfmplayground.configs.models import KumoClassifierConfig, KumoRegressorConfig
from tfmplayground.configs.training import ClassificationExperimentConfig, RegressionExperimentConfig
from tfmplayground.interface import TabularClassifier, TabularRegressor
from tfmplayground.models.kumo import KumoModel
from tfmplayground.utils import Experiment, load_model

TINY = dict(
    cell_channels=16,
    num_embedding_layers=1,
    num_embedding_heads=2,
    num_inducing_points=8,
    num_frequencies=4,
    num_readout_tokens=2,
    icl_channels=32,
    num_icl_layers=1,
    num_icl_heads=2,
)
CONFIGS = [
    KumoClassifierConfig(**TINY, num_classes=3),
    KumoRegressorConfig(**TINY, num_quantiles=9),
]


def batch(config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    gives random train rows, targets and test rows for config's problem
    """
    torch.manual_seed(0)
    x = torch.randn(2, 30, 5)
    y = torch.randint(0, 3, (2, 20)).float() if config.problem == "classification" else torch.randn(2, 20)
    return x[:, :20], y, x[:, 20:]


@pytest.mark.parametrize("config", CONFIGS)
def test_forward_gives_test_row_outputs(config):
    """Forward should give finite float32 outputs for every test row, in train and in eval mode."""
    model = KumoModel(config)
    num_outputs = model.borders.numel() - 1
    for training in (True, False):
        model.train(training)
        with torch.set_grad_enabled(training):
            output = model(*batch(config))

        assert output.shape == (2, 10, num_outputs) and output.dtype == torch.float32
        assert torch.isfinite(output).all()


def test_interfaces_predict():
    """Classifier probabilities should sum to 1 and regressor should give one value per test row."""
    rng = np.random.default_rng(0)
    X_train, X_test = rng.standard_normal((20, 4)), rng.standard_normal((10, 4))

    classifier = TabularClassifier(KumoModel(CONFIGS[0]), device="cpu").fit(X_train, rng.integers(0, 3, size=20))
    np.testing.assert_allclose(classifier.predict_proba(X_test).sum(axis=1), 1.0, rtol=1e-5)

    regressor = TabularRegressor(KumoModel(CONFIGS[1]), device="cpu").fit(X_train, rng.standard_normal(20))
    assert regressor.predict(X_test).shape == (10,)


@pytest.mark.parametrize(
    "config, experiment",
    [(CONFIGS[0], ClassificationExperimentConfig), (CONFIGS[1], RegressionExperimentConfig)],
)
def test_checkpoint_round_trip(config, experiment, tmp_path):
    """A reloaded checkpoint should give the same eval output."""
    model = KumoModel(config).eval()
    run = Experiment(experiment(experiments_dir=str(tmp_path)))
    path = tmp_path / "model.pth"
    run.save_checkpoint(path, model)
    reloaded = load_model(path).eval()

    with torch.no_grad():
        torch.testing.assert_close(reloaded(*batch(config)), model(*batch(config)))
