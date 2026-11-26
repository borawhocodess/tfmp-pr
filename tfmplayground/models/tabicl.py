import torch
from tabicl._model.inference_config import InferenceConfig
from tabicl._model.ssmax import QASSMaxMLP
from tabicl._model.tabicl import TabICL

from tfmplayground.configs.models import TabICLClassifierConfig, TabICLRegressorConfig
from tfmplayground.models.base import TabularFoundationModel


class Float32QASSMaxMLP(QASSMaxMLP):
    """
    qassmax that runs in float32 under autocast, since its scales grow in bfloat16 until training does not recover
    """

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        """
        scales queries in float32 and gives them back in their dtype
        """
        with torch.autocast(q.device.type, enabled=False):
            return super().forward(q.float(), n).to(q.dtype)


class TabICLModel(TabICL, TabularFoundationModel):
    """
    adapts tabicl to tabularfoundationmodel interface
    """

    io_modules = (
        "col_embedder.in_linear",
        "col_embedder.y_encoder",
        "icl_predictor.y_encoder",
        "icl_predictor.decoder",
    )

    def __init__(self, config: TabICLClassifierConfig | TabICLRegressorConfig) -> None:
        """
        builds tabicl from config and reserves its borders
        """
        self.config = config
        super().__init__(
            max_classes=config.max_classes,
            num_quantiles=config.num_quantiles,
            embed_dim=config.embed_dim,
            col_num_blocks=config.col_num_blocks,
            col_nhead=config.col_nhead,
            col_num_inds=config.col_num_inds,
            col_affine=config.col_affine,
            col_feature_group=config.col_feature_group,
            col_feature_group_size=config.col_feature_group_size,
            col_target_aware=config.col_target_aware,
            col_ssmax=config.col_ssmax,
            row_num_blocks=config.row_num_blocks,
            row_nhead=config.row_nhead,
            row_num_cls=config.row_num_cls,
            row_rope_base=config.row_rope_base,
            row_rope_interleaved=config.row_rope_interleaved,
            icl_num_blocks=config.icl_num_blocks,
            icl_nhead=config.icl_nhead,
            icl_ssmax=config.icl_ssmax,
            ff_factor=config.ff_factor,
            dropout=config.dropout,
            activation=config.activation,
            norm_first=config.norm_first,
            bias_free_ln=config.bias_free_ln,
            zero_init=config.zero_init,
            recompute=config.recompute,
        )
        self.register_buffer("borders", torch.zeros(config.num_quantiles + 1))
        for module in self.modules():
            if type(module) is QASSMaxMLP:
                module.__class__ = Float32QASSMaxMLP

    def hidden_matrices(self) -> list[torch.nn.Parameter]:
        """
        gives hidden matrices without the qassmax mlps, which schedule-free adamw trains instead of muon
        """
        qassmax = {
            id(parameter)
            for module in self.modules()
            if isinstance(module, QASSMaxMLP)
            for parameter in module.parameters()
        }
        return [parameter for parameter in super().hidden_matrices() if id(parameter) not in qassmax]

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        takes train rows as context and predicts test rows through tabicl forward

        joins train and test rows and sets the model device for native inference
        """
        X = torch.cat([X_train, X_test], dim=1)
        if self.training:
            return super().forward(X, y_train)
        config = InferenceConfig()
        device = next(self.parameters()).device
        for stage in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
            stage.device = device
        return super().forward(X, y_train, inference_config=config).float()
