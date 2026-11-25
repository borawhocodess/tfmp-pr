import math
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
    fit_borders: bool = True,
) -> nn.Module:
    """
    gives loss that matches problem

    fits bucket borders from prior when regression head needs them, unless model already has its own
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
            if fit_borders:
                model.borders = make_bucket_borders(
                    prior=prior,
                    num_buckets=num_outputs,
                    batch_size=training.batch_size,
                    min_targets=training.bucket_borders_min_targets,
                ).to(device)
        model.config.head = head
        return make_regression_decoder(model).to(device)


class MuonScheduleFree:
    """
    muon for hidden matrices and schedule-free adamw for every other parameter, with learning rate and weight decay
    shared
    """

    def __init__(self, model: TabularFoundationModel, lr: float, weight_decay: float) -> None:
        """
        splits parameters of model between muon and schedule-free adamw
        """
        hidden = model.hidden_matrices()
        hidden_ids = {id(parameter) for parameter in hidden}
        rest = [parameter for parameter in model.parameters() if id(parameter) not in hidden_ids]
        self.muon = torch.optim.Muon(hidden, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
        self.adamw = schedulefree.AdamWScheduleFree(rest, lr=lr, weight_decay=weight_decay)

    @property
    def param_groups(self) -> list[dict]:
        """
        gives parameter groups of both optimizers, so a schedule can set their learning rate
        """
        return [*self.muon.param_groups, *self.adamw.param_groups]

    def step(self) -> None:
        """
        takes one step of both optimizers
        """
        self.muon.step()
        self.adamw.step()

    def zero_grad(self) -> None:
        """
        clears gradients of both optimizers
        """
        self.muon.zero_grad()
        self.adamw.zero_grad()

    def train(self) -> None:
        """
        puts training weights into parameters of schedule-free adamw, muon has only one set
        """
        self.adamw.train()

    def eval(self) -> None:
        """
        puts averaged weights into parameters of schedule-free adamw, muon has only one set
        """
        self.adamw.eval()

    def state_dict(self) -> list[dict]:
        """
        gives states of both optimizers, so a run can resume
        """
        return [self.muon.state_dict(), self.adamw.state_dict()]

    def load_state_dict(self, state: list[dict]) -> None:
        """
        puts back states of both optimizers
        """
        self.muon.load_state_dict(state[0])
        self.adamw.load_state_dict(state[1])


def default_optimizer(
    model: TabularFoundationModel, training: TrainingConfig
) -> schedulefree.AdamWScheduleFree | MuonScheduleFree:
    """
    gives schedule-free adamw, or muon for hidden matrices with schedule-free adamw for the rest
    """
    if training.muon:
        return MuonScheduleFree(model, lr=training.lr, weight_decay=training.weight_decay)
    return schedulefree.AdamWScheduleFree(model.parameters(), lr=training.lr, weight_decay=training.weight_decay)


def learning_rate(training: TrainingConfig, step: int) -> float:
    """
    gives learning rate for step: linear warmup over warmup_proportion of steps, then constant, or decaying to 0 on a
    cosine over the remaining steps, or wsd: constant until the last decay_proportion of steps, then linear to 0
    """
    warmup_steps = training.warmup_proportion * training.steps
    if step < warmup_steps:
        return training.lr * step / warmup_steps
    if training.lr_schedule == "cosine":
        progress = (step - warmup_steps) / (training.steps - warmup_steps)
        return training.lr * 0.5 * (1 + math.cos(math.pi * progress))
    decay_steps = training.decay_proportion * training.steps
    if training.lr_schedule == "wsd" and step > training.steps - decay_steps:
        return training.lr * (training.steps - step) / decay_steps
    return training.lr


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
    callback = default_callback(problem, experiment, eval, device)

    if training.steps < 1:
        raise ValueError("steps must be at least 1")
    if training.eval_every < 1:
        raise ValueError("eval_every must be at least 1")
    if training.lr_schedule not in ("constant", "cosine", "wsd"):
        raise ValueError(f"{training.lr_schedule!r} learning rate schedule is not in (constant, cosine, wsd)")
    if not 0 <= training.warmup_proportion < 1:
        raise ValueError(f"warmup_proportion must be in [0, 1), not {training.warmup_proportion}")
    if training.lr_schedule == "wsd" and not 0 < training.decay_proportion <= 1 - training.warmup_proportion:
        raise ValueError(f"decay_proportion must be in (0, 1 - warmup_proportion], not {training.decay_proportion}")
    model.to(device)
    optimizer = default_optimizer(model, training)
    classification_task = problem == "classification"
    regression_task = not classification_task
    batches = iter(PriorDataLoader(prior, training.batch_size))
    step = 0
    if experimentconfig.resume is not None:
        step = experiment.load_resume(model, optimizer, prior)
        experiment.log(f"resumed at step {step}", console=True)
    criterion = default_criterion(
        problem, model, prior, training, device, fit_borders=experimentconfig.resume is None
    ).to(device)
    experiment.log_configs(model=model.config, prior=prior.config, eval=eval, training=training)

    try:
        while step < training.steps:
            train_start_time = time.time()
            model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            num_valid = 0
            while num_valid < training.eval_every and step < training.steps:
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
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(training, step)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.cpu().detach().item()
                num_valid += 1
                step += 1

            end_time = time.time()
            mean_loss = total_loss / num_valid
            model.eval()
            optimizer.eval()

            callback.on_eval(step, end_time - train_start_time, mean_loss, model)
            experiment.save_checkpoints(model)
            experiment.save_resume(model, optimizer, prior, step)
    except KeyboardInterrupt:
        pass
    finally:
        callback.close()

    return model
