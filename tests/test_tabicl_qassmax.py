import torch
from tabicl._model.ssmax import QASSMaxMLP

from tfmplayground.configs.models import TabICLClassifierConfig
from tfmplayground.models.tabicl import TabICLModel


def small_model() -> TabICLModel:
    return TabICLModel(TabICLClassifierConfig(embed_dim=32, col_num_inds=8, row_num_cls=2, icl_num_blocks=2))


def test_qassmax_weights_are_not_muon_matrices():
    """Muon should not get the qassmax mlp weights, so adamw trains them, but should still get the other matrices."""
    model = small_model()
    qassmax = {id(p) for m in model.modules() if isinstance(m, QASSMaxMLP) for p in m.parameters()}
    hidden = {id(p) for p in model.hidden_matrices()}

    assert qassmax
    assert hidden
    assert not qassmax & hidden


def test_qassmax_runs_in_float32_under_autocast():
    """Under bfloat16 autocast, qassmax should give the float32 result in bfloat16."""
    torch.manual_seed(0)
    layer = next(m for m in small_model().modules() if isinstance(m, QASSMaxMLP))
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter)
    q = torch.randn(2, layer.num_heads, 5, layer.head_dim).to(torch.bfloat16)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        scaled = layer(q, 1000)
    expected = QASSMaxMLP.forward(layer, q.float(), 1000).to(torch.bfloat16)

    assert scaled.dtype == torch.bfloat16
    assert torch.equal(scaled, expected)
