from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tfmplayground import models
from tfmplayground.configs import models as model_configs
from tfmplayground.configs.training import ClassificationExperimentConfig, ClassificationTrainingConfig
from tfmplayground.priors.base import Prior
from tfmplayground.training.callbacks import ClassifierExperimentEvaluationCallback
from tfmplayground.training.pretrain import pretrainTFM

TINY_CONFIGS = {
    "NanoTabPFN": dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2),
    "ModdedNanoTabPFN": dict(l=2, a=2, e=16, h=32, thinking_rows=4),
    "NanoTabICL": dict(
        embed_dim=16, col_num_blocks=1, row_num_blocks=1, icl_num_blocks=1, col_nhead=2, row_nhead=2, icl_nhead=2
    ),
    "TabICL": dict(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
    ),
    "TabFM": dict(),
}


def outputs_and_gradients(name: str, checkpointed: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """
    builds tiny model, runs one forward and backward pass, and gives output, gradient of every parameter and number of
    layer calls inside blocks during the backward pass
    """
    torch.manual_seed(0)
    config = getattr(model_configs, f"{name}ClassifierConfig")(**TINY_CONFIGS[name])
    model = getattr(models, f"{name}Model")(config=config)
    if checkpointed:
        model.checkpoint_blocks()
    model.train()
    calls = []
    for module in model.block_modules:
        for layer in model.get_submodule(module).modules():
            layer.register_forward_pre_hook(lambda layer, args: calls.append(1))
    x = torch.randn(2, 30, 5)
    y = torch.randint(0, 2, (2, 20)).float()
    output = model(x[:, :20], y, x[:, 20:])
    forward_calls = len(calls)
    output.float().square().mean().backward()
    gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    return output.detach(), gradients, len(calls) - forward_calls


@pytest.mark.parametrize("name", TINY_CONFIGS)
def test_block_modules_hold_blocks(name):
    """Every named block list should exist and hold blocks whose forward is then checkpointed."""
    config = getattr(model_configs, f"{name}ClassifierConfig")(**TINY_CONFIGS[name])
    model = getattr(models, f"{name}Model")(config=config)
    model.checkpoint_blocks()

    assert model.block_modules
    for module in model.block_modules:
        blocks = model.get_submodule(module)
        assert len(blocks) > 0, module
        assert all(block.forward.func is checkpoint for block in blocks), module


@pytest.mark.parametrize("name", TINY_CONFIGS)
def test_checkpointed_blocks_give_same_outputs_and_gradients(name):
    """Blocks should run again in the backward pass, without changing outputs or gradients."""
    output, gradients, backward_calls = outputs_and_gradients(name, checkpointed=False)
    checkpointed_output, checkpointed_gradients, checkpointed_backward_calls = outputs_and_gradients(
        name, checkpointed=True
    )

    assert backward_calls == 0
    assert checkpointed_backward_calls > 0

    torch.testing.assert_close(checkpointed_output, output)
    assert checkpointed_gradients.keys() == gradients.keys()
    for parameter, gradient in gradients.items():
        torch.testing.assert_close(checkpointed_gradients[parameter], gradient, msg=parameter)


class RandomPrior(Prior):
    """
    gives random tables from global torch randomness
    """

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, 30, 4)
        y = torch.randint(0, 3, (batch_size, 30)).float()
        return x[:, :20], y[:, :20], x[:, 20:], y[:, 20:]


def test_training_checkpoints_blocks_once_and_gives_them_back(tmp_path, monkeypatch):
    """Training with gradient_checkpointing should wrap blocks once per run and unwrap them after."""
    forwards = []

    def evaluate(self, model, **kwargs):
        forwards.extend(block.forward for block in model.transformer_blocks)
        return [0.5]

    monkeypatch.setattr(ClassifierExperimentEvaluationCallback, "evaluate", evaluate)
    config = model_configs.NanoTabPFNClassifierConfig(**TINY_CONFIGS["NanoTabPFN"], num_outputs=3)
    model = models.NanoTabPFNModel(config=config)
    for _ in range(2):
        model = pretrainTFM(
            "classification",
            model,
            RandomPrior(),
            training=ClassificationTrainingConfig(steps=2, eval_every=2, gradient_checkpointing=True),
            experiment=ClassificationExperimentConfig(experiments_dir=str(tmp_path)),
            device="cpu",
        )
        assert not any("forward" in vars(block) for block in model.transformer_blocks)

    assert len(forwards) == 2 * len(model.transformer_blocks)
    assert all(forward.func is checkpoint and not isinstance(forward.args[0], partial) for forward in forwards)
