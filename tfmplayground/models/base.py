from abc import ABC, abstractmethod
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TabularFoundationModel(nn.Module, ABC):
    """
    base class for every model this package trains

    block_modules names lists of transformer blocks, whose activations gradient checkpointing recomputes
    """

    block_modules: tuple[str, ...] = ()

    def checkpoint_blocks(self) -> None:
        """
        makes every block recompute its activations in the backward pass instead of keeping them, which saves memory
        and costs one more forward pass per block
        """
        for name in self.block_modules:
            for block in self.get_submodule(name):
                block.forward = partial(checkpoint, block.forward, use_reentrant=False)

    def uncheckpoint_blocks(self) -> None:
        """
        gives every block its own forward back, which undoes checkpoint_blocks
        """
        for name in self.block_modules:
            for block in self.get_submodule(name):
                vars(block).pop("forward", None)

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
