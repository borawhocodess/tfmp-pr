from abc import ABC, abstractmethod

import torch
from torch import nn


def standardize_features(x: torch.Tensor, num_train_rows: int) -> torch.Tensor:
    """
    standardizes features with mean and deviation of train rows
    """
    train_rows = x[:, :num_train_rows]
    mean = train_rows.mean(dim=1, keepdim=True)
    std = train_rows.std(dim=1, correction=0, keepdim=True) + 1e-8
    return (x - mean) / std


class TabularFoundationModel(nn.Module, ABC):
    """
    base class for every model this package trains
    """

    @abstractmethod
    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        predicts test rows with train rows as context
        """
        ...
