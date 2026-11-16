from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrainingConfig(ABC):
    """
    settings training runs share
    """

    seed: int = 2402
    lr: float = 1e-4
    batch_size: int = 1
    steps: int = 1_000_000
    eval_every: int = 100
    grad_clip: float = 1.0
    mixed_precision: bool = False
    prior_workers: int = 0
    muon: bool = False
    weight_decay: float = 0.0
    lr_schedule: str = "constant"
    warmup_proportion: float = 0.0
    accumulation_steps: int = 1

    @property
    @abstractmethod
    def problem(self) -> str: ...


@dataclass
class ClassificationTrainingConfig(TrainingConfig):
    """
    training settings for classification
    """

    problem: str = "classification"


@dataclass
class RegressionTrainingConfig(TrainingConfig):
    """
    training settings for regression
    """

    problem: str = "regression"
    criterion: str | None = None
    bucket_borders_min_targets: int = 1_000_000


@dataclass
class ExperimentConfig(ABC):
    """
    settings experiment tracking shares
    """

    name: str = "test"
    experiments_dir: str = "workdir/experiments"
    resume: str | None = None

    @property
    @abstractmethod
    def problem(self) -> str: ...


@dataclass
class ClassificationExperimentConfig(ExperimentConfig):
    """
    experiment settings for classification
    """

    problem: str = "classification"


@dataclass
class RegressionExperimentConfig(ExperimentConfig):
    """
    experiment settings for regression
    """

    problem: str = "regression"
