from abc import ABC, abstractmethod

import torch
from torch import nn


class TabularFoundationModel(nn.Module, ABC):
    """
    base class for every model this package trains

    io_modules names input and output modules, which muon leaves to adamw
    """

    io_modules: tuple[str, ...] = ()

    def hidden_matrices(self) -> list[nn.Parameter]:
        """
        gives weight matrices outside of input and output modules, which muon optimizes
        """
        return [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.ndim == 2
            and name.endswith("weight")
            and not any(name.startswith(f"{module}.") for module in self.io_modules)
        ]

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
