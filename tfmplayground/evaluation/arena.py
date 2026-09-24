"""
official tabarena and beyondarena evaluation of pretrained checkpoints

wraps our models as autogluon models, so tabarena runs its own pipeline on them:
its tasks and splits, its preprocessing, its validation protocol and its metrics,
and compares the result against its leaderboard
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

from tfmplayground.interface import TabularClassifier, TabularRegressor
from tfmplayground.utils import load_model

ARENAS = ("tabarena", "beyondarena")
# model configs name how many classes they predict differently, some have more than one of these
CLASS_LIMIT_KEYS = ("max_classes", "out_dim", "num_outputs", "o")


class TFMPlaygroundModel(AbstractTorchModel):
    """
    autogluon model that does in-context learning with a pretrained checkpoint

    lives in an importable module, not __main__, so ray workers can unpickle it
    """

    ag_key = "TFMP"
    ag_name = "TFMPlayground"
    ag_priority = 65

    default_num_gpus = 1
    default_resources_physical_cores_only = True
    gpu_strongly_recommended = True

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int | float = 0,
        **kwargs,
    ) -> None:
        """
        loads checkpoint and stores training rows as context
        """
        params = self._get_model_params()
        checkpoint_path = params.get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("TFMPlaygroundModel needs a checkpoint_path hyperparameter")
        model = load_model(checkpoint_path)
        problem = "regression" if self.problem_type == "regression" else "classification"
        if model.config.problem != problem:
            raise ValueError(f"checkpoint does {model.config.problem!r}, task needs {problem!r}")

        device = self._resolve_fit_device(num_gpus=num_gpus)
        X = self.preprocess(X)
        y = np.asarray(y)
        if problem == "classification":
            self.model = TabularClassifier(model, device=device)
        else:
            self.model = TabularRegressor(model, device=device)
            y = y.astype(np.float64)
        self.model.fit(X, y)

    def get_device(self) -> str:
        """
        device of fitted network
        """
        return str(self.model.device)

    def _set_device(self, device: str) -> None:
        """
        moves fitted network, inputs follow in predict
        """
        self.model.model.to(device)
        self.model.device = device

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        """
        fits folds one after another, then refits on all data, like other foundation models in tabarena
        """
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update({"fold_fitting_strategy": "sequential_local", "refit_folds": True})
        return default_ag_args_ensemble

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


def make_context(arena: str, **kwargs):
    """
    gives tabarena context that knows arena tasks, baselines and protocol
    """
    if arena == "tabarena":
        from tabarena.contexts import TabArenaContext

        return TabArenaContext(**kwargs)
    if arena == "beyondarena":
        from tabarena.contexts import BeyondArenaContext

        return BeyondArenaContext(**kwargs)
    raise ValueError(f"{arena!r} arena is not in {ARENAS}")


def make_bundle(arena: str, **kwargs):
    """
    gives experiment bundle that matches arena
    """
    if arena == "tabarena":
        from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle

        return TabArenaV0pt1ExperimentBundle(**kwargs)
    if arena == "beyondarena":
        from tabarena.benchmark.experiment import BeyondArenaExperimentBundle

        return BeyondArenaExperimentBundle(**kwargs)
    raise ValueError(f"{arena!r} arena is not in {ARENAS}")


def default_subset(arena: str, problem: str) -> list[str]:
    """
    gives recommended split subset of arena, restricted to problem of checkpoint
    """
    splits = "lite" if arena == "tabarena" else "core"
    return [splits, problem]


def checkpoint_max_classes(checkpoint_path: str | Path) -> int | None:
    """
    gives how many classes checkpoint can predict, None for regression

    smallest of all class limits in config, e.g. nanotabicl embeds max_classes but emits out_dim logits
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["problem"] != "classification":
        return None
    model_config = checkpoint["model_config"]
    limits = [model_config[key] for key in CLASS_LIMIT_KEYS if key in model_config]
    if limits:
        return min(limits)
    raise ValueError(f"{checkpoint['config_class']} has none of {CLASS_LIMIT_KEYS}, cannot tell its class limit")


def make_experiments(
    checkpoint_path: str | Path,
    *,
    arena: str = "tabarena",
    outer: bool = False,
    max_n_features: int | None = 500,
    max_n_samples: int | None = 10_000,
    max_n_classes: int | None = None,
    time_limit: int | None = None,
    num_gpus: int | None = None,
) -> list:
    """
    gives arena experiments that run checkpoint under arena settings

    tasks beyond what model supports are skipped, None means no limit,
    class limit is never above what checkpoint can predict
    """
    from tabarena.benchmark.experiment.model_constraints import ModelConstraints
    from tabarena.utils.config_utils import ConfigGenerator

    config_generator = ConfigGenerator(
        model_cls=TFMPlaygroundModel,
        manual_configs=[{"checkpoint_path": str(Path(checkpoint_path).resolve())}],
        search_space={},
    )
    checkpoint_classes = checkpoint_max_classes(checkpoint_path)
    if checkpoint_classes is not None:
        max_n_classes = checkpoint_classes if max_n_classes is None else min(max_n_classes, checkpoint_classes)
    constraints = ModelConstraints(
        max_n_features=max_n_features,
        max_n_samples_train_per_fold=max_n_samples,
        max_n_classes=max_n_classes,
    )
    bundle = make_bundle(
        arena,
        models=[(config_generator, 0)],
        outer_experiments=outer,
        custom_model_constraints={TFMPlaygroundModel.ag_key: constraints},
    )
    return bundle.build_experiments(time_limit=time_limit, num_gpus=num_gpus)


def evaluate_arena(
    checkpoint_path: str | Path,
    *,
    arena: str = "tabarena",
    subset: str | list[str] | None = None,
    dataset_names: list[str] | None = None,
    outer: bool = False,
    max_n_features: int | None = 500,
    max_n_samples: int | None = 10_000,
    max_n_classes: int | None = None,
    time_limit: int | None = None,
    num_gpus: int | None = None,
    results_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    name: str = "TFMP",
    debug_mode: bool = True,
) -> pd.DataFrame:
    """
    runs official arena evaluation on checkpoint and compares it against leaderboard

    Parameters
    ----------
    checkpoint_path : str | Path
        checkpoint saved during pretraining
    arena : str
        tabarena or beyondarena
    subset : str | list[str], optional
        arena subset expression, defaults to lite (tabarena) or core (beyondarena) tasks of checkpoint problem
    dataset_names : list[str], optional
        restricts evaluation to these datasets
    outer : bool
        fits once on all training data instead of official bagging protocol, faster but unofficial
    max_n_features, max_n_samples, max_n_classes : int, optional
        tasks beyond what model supports are skipped, None means no limit,
        class limit is never above what checkpoint can predict
    time_limit : int, optional
        fit time limit per task in seconds, defaults to arena default
    num_gpus : int, optional
        gpus per fit, defaults to what machine has
    results_dir : str | Path, optional
        cache for raw results, so reruns resume, defaults to throwaway directory
    output_dir : str | Path, optional
        where leaderboard figures and tables go, nothing is written when None
    name : str
        prefix that marks our method in leaderboard
    debug_mode : bool
        runs sequentially in this process instead of ray workers

    Returns
    -------
    pd.DataFrame
        arena leaderboard including our method
    """
    if subset is None:
        problem = torch.load(checkpoint_path, map_location="cpu")["problem"]
        subset = default_subset(arena, problem)
    experiments = make_experiments(
        checkpoint_path,
        arena=arena,
        outer=outer,
        max_n_features=max_n_features,
        max_n_samples=max_n_samples,
        max_n_classes=max_n_classes,
        time_limit=time_limit,
        num_gpus=num_gpus,
    )
    context = make_context(arena)
    build_kwargs = {"dataset_names": dataset_names} if dataset_names is not None else None
    context.build_and_run_jobs(
        experiments,
        expname=str(results_dir) if results_dir is not None else None,
        subset=subset,
        build_kwargs=build_kwargs,
        new_result_prefix=f"[{name}] ",
        debug_mode=debug_mode,
    )
    return context.compare(output_dir=Path(output_dir) if output_dir is not None else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluates pretrained checkpoint on tabarena or beyondarena")
    parser.add_argument("checkpoint", type=str, help="checkpoint saved during pretraining")
    parser.add_argument("--arena", type=str, default="tabarena", choices=ARENAS)
    parser.add_argument("--subset", type=str, nargs="+", default=None, help="arena subset expressions")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="restricts to these datasets")
    parser.add_argument("--outer", action="store_true", help="single fit on all training data, no bagging")
    parser.add_argument("--max_n_features", type=int, default=500)
    parser.add_argument("--max_n_samples", type=int, default=10_000)
    parser.add_argument("--max_n_classes", type=int, default=None, help="defaults to checkpoint class limit")
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default="TFMP")
    parser.add_argument("--ray", action="store_true", help="runs jobs on ray workers")
    args = parser.parse_args()

    leaderboard = evaluate_arena(
        args.checkpoint,
        arena=args.arena,
        subset=args.subset,
        dataset_names=args.datasets,
        outer=args.outer,
        max_n_features=args.max_n_features,
        max_n_samples=args.max_n_samples,
        max_n_classes=args.max_n_classes,
        time_limit=args.time_limit,
        num_gpus=args.num_gpus,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        name=args.name,
        debug_mode=not args.ray,
    )
    print(leaderboard.to_markdown())


if __name__ == "__main__":
    main()
