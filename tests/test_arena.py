from dataclasses import asdict

import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tfmplayground.configs import models as model_configs
from tfmplayground.configs.models import NanoTabPFNClassifierConfig, NanoTabPFNRegressorConfig
from tfmplayground.configs.training import ClassificationExperimentConfig
from tfmplayground.evaluation.arena import CLASS_LIMIT_KEYS, make_experiments
from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import Experiment

pytest.importorskip("tabarena")

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle  # noqa: E402
from tabarena.benchmark.task import UserTask  # noqa: E402
from tabarena.benchmark.task.metadata import TaskMetadataCollection  # noqa: E402
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits  # noqa: E402
from tabarena.contexts import TABARENA_V0PT1_VALIDATION_PROTOCOL, AbstractArenaContext  # noqa: E402

SIZES = dict(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2)


def save_checkpoint(tmp_path, model, name):
    experiment = Experiment(ClassificationExperimentConfig(experiments_dir=str(tmp_path / "experiments")))
    path = tmp_path / name
    experiment.save_checkpoint(path, model)
    return path


def toy_frame(classification: bool) -> pd.DataFrame:
    """small table with numeric, categorical and missing features"""
    maker = make_classification if classification else make_regression
    X, y = maker(n_samples=120, n_features=6, random_state=0)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df.loc[::9, "num_0"] = float("nan")
    df["cat"] = pd.Categorical(["a", "b", "c"] * 40)
    return df.assign(target=y)


def make_task(tmp_path, classification: bool):
    dataset = toy_frame(classification)
    if classification:
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
        splits = from_sklearn_splits_to_user_task_splits(
            folds.split(dataset.drop(columns="target"), dataset["target"]), n_splits=2
        )
    else:
        train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.3, random_state=0)
        splits = {0: {0: (train_indices, test_indices)}}
    problem = "classification" if classification else "regression"
    task = UserTask(task_name=f"toy_{problem}", task_cache_path=tmp_path / "tasks")
    wrapper = task.create_task(dataset=dataset, target_feature="target", problem_type=problem, splits=splits)
    task.save_task(wrapper)
    return task, wrapper.metadata


@pytest.mark.parametrize(
    "config_class",
    [getattr(model_configs, name) for name in dir(model_configs) if name.endswith("ClassifierConfig")],
)
def test_classifier_configs_expose_class_limit(config_class):
    """every classifier checkpoint tells how many classes it can predict"""
    assert any(key in asdict(config_class()) for key in CLASS_LIMIT_KEYS)


@pytest.mark.parametrize(("requested", "expected"), [(None, 3), (10, 3), (2, 2)])
def test_class_limit_follows_checkpoint(tmp_path, requested, expected):
    """tasks with more classes than checkpoint predicts are never scheduled"""
    model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(num_outputs=3, **SIZES))
    checkpoint_path = save_checkpoint(tmp_path, model, "classification.pth")
    experiments = make_experiments(checkpoint_path, max_n_classes=requested)
    assert [experiment.model_constraints.max_n_classes for experiment in experiments] == [expected]


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_tabarena_pipeline_scores_checkpoint(tmp_path, problem, outer):
    """a checkpoint runs through tabarena's own protocol and gets a leaderboard entry per task"""
    torch.manual_seed(0)
    if problem == "classification":
        model = NanoTabPFNModel(config=NanoTabPFNClassifierConfig(num_outputs=3, **SIZES))
    else:
        model = NanoTabPFNModel(config=NanoTabPFNRegressorConfig(**SIZES))
        model.borders = torch.linspace(-3, 3, model.borders.numel())
    checkpoint_path = save_checkpoint(tmp_path, model, f"{problem}.pth")

    task, metadata = make_task(tmp_path, classification=problem == "classification")
    context = AbstractArenaContext(
        task_metadata=TaskMetadataCollection.from_source([metadata]),
        methods=[],
        validation_protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL,
        backend="native",
    )
    experiments = make_experiments(checkpoint_path, outer=outer, time_limit=600, num_gpus=0)
    baselines = TabArenaV0pt1ExperimentBundle(models=[("Linear", 0)], outer_experiments=outer)
    experiments += baselines.build_experiments(time_limit=600, num_gpus=0)
    results = context.build_and_run_jobs(
        experiments,
        expname=None,
        user_tasks=[task],
        new_result_prefix="[New] ",
        debug_mode=True,
    )

    n_splits = 2 if problem == "classification" else 1
    assert len(results) == 2 * n_splits
    leaderboard = context.compare(output_dir=None)
    ours = [method for method in leaderboard["method"] if "TFMP" in method]
    assert len(ours) == 1
