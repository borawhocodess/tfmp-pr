import time

import schedulefree
import torch
from torch import nn

from tfmplayground.configs.evaluation import EvaluationConfig
from tfmplayground.configs.training import (
    ClassificationExperimentConfig,
    ClassificationTrainingConfig,
    ExperimentConfig,
    RegressionExperimentConfig,
    RegressionTrainingConfig,
    TrainingConfig,
)
from tfmplayground.models.base import TabularFoundationModel
from tfmplayground.priors.base import Prior, PriorDataLoader
from tfmplayground.training.callbacks import (
    ClassifierExperimentEvaluationCallback,
    ExperimentEvaluationCallback,
    RegressorExperimentEvaluationCallback,
)
from tfmplayground.utils import (
    Experiment,
    get_default_device,
    make_bucket_borders,
    make_regression_decoder,
    set_randomness_seed,
)


def default_training(problem: str) -> TrainingConfig:
    """
    gives training config that matches problem
    """
    if problem == "classification":
        return ClassificationTrainingConfig()
    if problem == "regression":
        return RegressionTrainingConfig()


def default_experiment(problem: str) -> ExperimentConfig:
    """
    gives experiment config that matches problem
    """
    if problem == "classification":
        return ClassificationExperimentConfig()
    if problem == "regression":
        return RegressionExperimentConfig()


def default_callback(
    problem: str,
    experiment: Experiment,
    config: EvaluationConfig,
    device: str | torch.device,
) -> ExperimentEvaluationCallback:
    """
    gives evaluation callback that matches problem
    """
    if problem == "classification":
        return ClassifierExperimentEvaluationCallback(experiment, config=config, device=device)
    if problem == "regression":
        return RegressorExperimentEvaluationCallback(experiment, config=config, device=device)


def check_problems(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    training: TrainingConfig | None,
    experiment: ExperimentConfig | None,
) -> None:
    """
    checks that problem is known and every config agrees with it
    """
    if problem not in ("classification", "regression"):
        raise ValueError(f"{problem!r} problem is not in (classification, regression)")
    for x in (model, prior, training, experiment):
        x_config = getattr(x, "config", x)
        x_problem = getattr(x_config, "problem", None)
        if x_problem is not None and x_problem != problem:
            raise ValueError(f"{type(x).__name__} must do {problem!r}, not {x_problem!r}")


def default_criterion(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    training: TrainingConfig,
    device: str | torch.device,
) -> nn.Module:
    """
    gives loss that matches problem

    fits bucket borders from prior when regression head needs them
    """
    if problem == "classification":
        return nn.CrossEntropyLoss()
    if problem == "regression":
        head = training.criterion if training.criterion is not None else model.config.head
        num_outputs = model.borders.numel() - 1
        if head == "scalar" and num_outputs != 1:
            raise ValueError(f"{head!r} head needs 1 output, not {num_outputs}")
        if head == "buckets":
            if num_outputs < 2:
                raise ValueError(f"{head!r} head needs more than 1 output, not {num_outputs}")
            model.borders = make_bucket_borders(
                prior=prior,
                num_buckets=num_outputs,
                batch_size=training.batch_size,
                min_targets=training.bucket_borders_min_targets,
            ).to(device)
        model.config.head = head
        return make_regression_decoder(model).to(device)


def pretrainTFM(
    problem: str,
    model: TabularFoundationModel,
    prior: Prior,
    eval: EvaluationConfig | None = None,
    training: TrainingConfig | None = None,
    experiment: ExperimentConfig | None = None,
    device: str | torch.device | None = None,
) -> TabularFoundationModel:
    """
    pretrains model on prior and gives it back trained

    batches with non-finite values are retried

    Parameters
    ----------
    problem : str
        classification or regression
    model : TabularFoundationModel
        model to pretrain
    prior : Prior
        prior that gives training tables
    eval : EvaluationConfig, optional
        tasks and their limits for periodic evaluations
    training : TrainingConfig, optional
        seed, sizes and limits for training loop
    experiment : ExperimentConfig, optional
        name and directory for experiment artifact tracking
    device : torch.device, optional
        device that holds model and training batches

    Returns
    -------
    TabularFoundationModel
        pretrained model ready for inference
    """
    check_problems(problem, model, prior, training, experiment)
    experimentconfig = experiment if experiment is not None else default_experiment(problem)
    experiment = Experiment(config=experimentconfig)
    training = training if training is not None else default_training(problem)
    set_randomness_seed(training.seed)
    device = device if device is not None else get_default_device()
    eval = eval if eval is not None else EvaluationConfig()
    criterion = default_criterion(problem, model, prior, training, device)
    callback = default_callback(problem, experiment, eval, device)
    experiment.log_configs(model=model.config, prior=prior.config, eval=eval, training=training)

    if training.steps < 1:
        raise ValueError("steps must be at least 1")
    model.to(device)
    criterion = criterion.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=training.lr, weight_decay=0.0)
    classification_task = problem == "classification"
    regression_task = not classification_task
    batches = iter(PriorDataLoader(prior, training.batch_size))

    try:
        for epoch in range(1, training.epochs + 1):
            epoch_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            num_valid = 0
            while num_valid < training.steps:
                x_train, y_train, x_test, y_test = next(batches)
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                if not all(torch.isfinite(t).all() for t in (x_train, y_train, x_test, y_test)):
                    continue

                if regression_task:
                    y_mean = y_train.mean(dim=1, keepdim=True)
                    y_std = y_train.std(dim=1, keepdim=True) + 1e-8
                    y_train = (y_train - y_mean) / y_std
                    y_test = (y_test - y_mean) / y_std
                    if not torch.isfinite(y_train).all() or not torch.isfinite(y_test).all():
                        continue

                output = model(x_train, y_train, x_test)
                if not torch.isfinite(output).all():
                    continue
                if classification_task:
                    y_test = y_test.reshape((-1,)).to(torch.long)
                    output = output.reshape(-1, output.shape[-1])
                    output_classes = output.shape[-1]
                    prior_classes = int(y_test.max()) + 1
                    if prior_classes > output_classes:
                        raise ValueError(f"model has {output_classes} outputs, prior gives {prior_classes} classes")

                losses = criterion(output, y_test)
                if not torch.isfinite(losses).all():
                    continue

                loss = losses.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.cpu().detach().item()
                num_valid += 1

            end_time = time.time()
            mean_loss = total_loss / num_valid
            model.eval()
            optimizer.eval()

            callback.on_epoch_end(epoch, end_time - epoch_start_time, mean_loss, model)
            experiment.save_checkpoints(model)
    except KeyboardInterrupt:
        pass
    finally:
        callback.close()

    return model
