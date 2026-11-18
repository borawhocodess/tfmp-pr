import torch
from sdm.models.kumo.tabular.model import _KumoTabular

from tfmplayground.configs.models import KumoClassifierConfig, KumoRegressorConfig
from tfmplayground.models.base import TabularFoundationModel


class KumoModel(_KumoTabular, TabularFoundationModel):
    """
    adapts kumo tabular to tabularfoundationmodel interface
    """

    def __init__(self, config: KumoClassifierConfig | KumoRegressorConfig) -> None:
        """
        builds kumo tabular network from config and reserves its borders
        """
        self.config = config
        super().__init__(
            num_classes=config.num_classes,
            num_quantiles=config.num_quantiles,
            cell_channels=config.cell_channels,
            num_embedding_layers=config.num_embedding_layers,
            num_embedding_heads=config.num_embedding_heads,
            num_inducing_points=config.num_inducing_points,
            group_size=config.group_size,
            num_frequencies=config.num_frequencies,
            num_readout_tokens=config.num_readout_tokens,
            icl_channels=config.icl_channels,
            num_icl_layers=config.num_icl_layers,
            num_icl_heads=config.num_icl_heads,
            num_icl_key_value_heads_for_query=config.num_icl_key_value_heads_for_query,
        )
        self.register_buffer("borders", torch.zeros((config.num_classes or config.num_quantiles) + 1))

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        takes train rows as context and predicts test rows through kumo tabular forward

        joins train and test rows, gives class targets as indices and marks no column as categorical
        """
        x = torch.cat([X_train, X_test], dim=1)
        y = y_train.long() if self.config.num_classes else y_train
        categorical_mask = torch.zeros(x.shape[0], x.shape[2], dtype=torch.bool, device=x.device)
        return super().forward(x, y, categorical_mask).float()
