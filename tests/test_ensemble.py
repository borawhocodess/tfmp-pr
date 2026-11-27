import numpy as np
import pytest
import torch

from tfmplayground.interface import TabularClassifier


class NearestNeighbor(torch.nn.Module):
    """
    predicts the class of the nearest train row as logits over 10 outputs, so any feature order or quantile transform
    keeps its answer and a wrong class shift changes it
    """

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        nearest = torch.cdist(X_test, X_train).argmin(dim=-1)
        labels = torch.gather(y_train, 1, nearest).long()
        return 10.0 * torch.nn.functional.one_hot(labels, 10).float()


class RecordingNearestNeighbor(NearestNeighbor):
    """
    nearest-neighbor model that records the classes of every context it gets
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen = []

    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        self.seen.append(set(y_train.long().flatten().tolist()))
        return super().forward(X_train, y_train, X_test)


@pytest.mark.parametrize("n_estimators, max_context", [(1, None), (4, None), (4, 60)])
def test_ensemble_keeps_classes(n_estimators, max_context):
    """Every estimator undoes its class shift, so the ensemble of a nearest-neighbor model predicts the true classes."""
    rng = np.random.default_rng(0)
    centers = 10 * rng.normal(size=(3, 5))
    y = rng.integers(0, 3, 120)
    X = centers[y] + rng.normal(size=(120, 5))
    classifier = TabularClassifier(NearestNeighbor(), "cpu", n_estimators=n_estimators, max_context=max_context)
    classifier.fit(X[:100], np.array(["a", "b", "c"])[y[:100]])
    assert (classifier.predict(X[100:]) == np.array(["a", "b", "c"])[y[100:]]).all()
    assert np.allclose(classifier.predict_proba(X[100:]).sum(axis=1), 1)
    X_test = classifier.feature_preprocessor.transform(X[100:])
    for i in range(n_estimators):
        assert (classifier.estimator_proba(X_test, i).argmax(axis=1) == y[100:]).all()


def test_context_subsets_keep_every_class_and_repeat():
    """A small max_context keeps a one-row minority class, and the same seed gives the same probabilities again."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20200, 3))
    y = np.zeros(20200, dtype=int)
    y[0] = 1
    classifier = TabularClassifier(NearestNeighbor(), "cpu", n_estimators=2, max_context=10)
    classifier.fit(X[:20000], y[:20000])
    X_test = classifier.feature_preprocessor.transform(X[20000:])
    classifier.model = RecordingNearestNeighbor()
    classifier.estimator_proba(X_test, 0)
    assert classifier.model.seen == [{0, 1}]
    classifier.max_context = None
    np.testing.assert_array_equal(classifier.predict_proba(X[20000:]), classifier.predict_proba(X[20000:]))
