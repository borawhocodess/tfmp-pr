from abc import ABC, abstractmethod
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TabularFoundationModel(nn.Module, ABC):
    """
    base class for every model this package trains

    io_modules names input and output modules, which muon leaves to adamw

    block_modules names lists of transformer blocks, whose activations gradient checkpointing recomputes
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

    block_modules: tuple[str, ...] = ()

    def checkpoint_blocks(self) -> None:
        """
        makes every block recompute its activations in the backward pass instead of keeping them, which saves memory
        and costs one more forward pass per block
        """
        for name in self.block_modules:
            for block in self.get_submodule(name):
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)

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
