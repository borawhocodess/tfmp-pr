# Vendored from https://github.com/NVIDIA/structured-data-models (train/kumo/tabular/prior.py, and
# _standardize_features and _robust_probe_target from train/kumo/tabular/loader.py) at c17f88f
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import torch.nn.functional as F
from torch import Tensor

import sdm.prior.tabular.transform as pt
from sdm import Task
from sdm.prior import (
    ActivationMatrixSampler,
    BallRootSampler,
    CallableActivation,
    Choice,
    ConcatParentMerger,
    Correlated,
    CorrelatedRootSampler,
    DistributionalMixtureConverter,
    GaussianMatrixSampler,
    IntegerPowActivation,
    InversePowActivation,
    KernelMatrixSampler,
    LogUniform,
    LogUniformInt,
    MixtureActivation,
    MixtureConverter,
    MixtureFunction,
    MixtureMatrixSampler,
    MixtureParentMerger,
    MixtureRootSampler,
    NearestNeighborConverter,
    NormalRootSampler,
    PowerLawMatrixSampler,
    RandomCauchyDAGSampler,
    RandomDiscretization,
    RandomEMAssignment,
    RandomGP,
    RandomLinear,
    RandomNN,
    RandomProduct,
    RandomQuadratic,
    RandomTree,
    ReduceParentMerger,
    ReLUPowActivation,
    SignPowActivation,
    SingularValuesMatrixSampler,
    SoftmaxCategoricalConverter,
    StandardizedActivation,
    Uniform,
    UniformInt,
    UniformRootSampler,
    argsort,
    gaussian,
    heaviside,
    identity,
    log_abs,
    modulo,
    one_hot_argmax,
    rank,
    softmax,
    unit_indicator,
)
from sdm.prior.tabular import SchemaSampler, SCMPrior


def make_prior(
    task_type: Task,
    *,
    correlated_block_prob: float = 0.3,
    relabel_prob: float = 0.1,
    coarsen_prob: float = 0.1,
    high_cardinality_prob: float = 0.05,
    y_heavy_tail_prob: float = 0.2,
    structured_missingness_prob: float = 0.2,
    mcar_missingness_prob: float = 0.01,
    near_empty_missingness_prob: float = 0.05,
    latent_driver_missingness_prob: float = 0.03,
    softmax_argmax_prob: float = 0.15,
) -> SCMPrior:
    """Build the prior used to train Kumo Tabular models.

    Args:
        task_type: Classification or regression task.
        correlated_block_prob: Probability of correlated feature blocks.
        relabel_prob: Probability of replacing the target.
        coarsen_prob: Probability of coarsening features.
        high_cardinality_prob: Probability of high-cardinality recoding.
        y_heavy_tail_prob: Probability of regression target warping.
        structured_missingness_prob: Structured masking probability.
        mcar_missingness_prob: Probability of 1-8% random cell masking.
        near_empty_missingness_prob: Probability of masking column groups
            to few observed rows.
        latent_driver_missingness_prob: Probability of masking driven by
            an unobserved row factor.
        softmax_argmax_prob: Probability of a softmax-converter call using
            bounded-bias argmax.
    """
    schema_sampler = SchemaSampler(
        task_type=task_type,
        num_columns=UniformInt(2, 100),
        cat_ratio=Uniform(-0.5, 1.2),
        max_cardinality=LogUniformInt(2, 200),
        num_classes=UniformInt(2, 10)
        if task_type == Task.classification
        else None,
        independent_cat_ratio=Uniform(-0.1, 1.1),
        max_correlated_cardinality=9,
    )
    dag_sampler = RandomCauchyDAGSampler(num_nodes=LogUniformInt(2, 32))
    latent_channels = Correlated(LogUniformInt(1, 32))
    root_sampler = MixtureRootSampler(
        samplers=[
            (NormalRootSampler(mean=0.0, std=1.0), 0.25),
            (UniformRootSampler(low=-1.0, high=1.0), 0.25),
            (BallRootSampler(), 0.25),
            (
                CorrelatedRootSampler(
                    MixtureRootSampler(
                        samplers=[
                            NormalRootSampler(mean=0.0, std=1.0),
                            UniformRootSampler(low=-1.0, high=1.0),
                        ],
                    )
                ),
                0.25,
            ),
        ],
    )
    fixed_activation = MixtureActivation(
        activations=[
            CallableActivation(function)
            for function in (
                Tensor.tanh,
                F.leaky_relu,
                F.elu,
                identity,
                F.selu,
                F.silu,
                Tensor.relu,
                F.softplus,
                F.relu6,
                F.hardtanh,
                Tensor.sign,
                heaviside,
                gaussian,
                Tensor.exp,
                unit_indicator,
                Tensor.sin,
                Tensor.square,
                Tensor.abs,
                softmax,
                one_hot_argmax,
                argsort,
                F.logsigmoid,
                log_abs,
                rank,
                Tensor.sigmoid,
                Tensor.round,
                modulo,
            )
        ],
        key="TorchActivation.kind",
    )
    node_activation = StandardizedActivation(
        MixtureActivation(
            activations=[
                (fixed_activation, 4.0),
                ReLUPowActivation(exponent=Correlated(LogUniform(0.1, 10.0))),
                SignPowActivation(exponent=Correlated(LogUniform(0.1, 10.0))),
            ],
        ),
        scale=Correlated(LogUniform(1.0, 10.0)),
    )

    matrix_sampler = MixtureMatrixSampler(
        samplers=[
            GaussianMatrixSampler(),
            PowerLawMatrixSampler(),
            SingularValuesMatrixSampler(),
            KernelMatrixSampler(
                num_centers=Correlated(
                    LogUniformInt(1, 15),
                    key="GaussianMixturePoints.num_centers",
                ),
                gamma=Correlated(
                    LogUniform(0.1, 10.0), key="KernelMatrix.scale"
                ),
            ),
            ActivationMatrixSampler(
                matrix_sampler=MixtureMatrixSampler(
                    samplers=[
                        GaussianMatrixSampler(normalize=False),
                        PowerLawMatrixSampler(normalize=False),
                        SingularValuesMatrixSampler(normalize=False),
                        KernelMatrixSampler(
                            num_centers=Correlated(
                                LogUniformInt(1, 15),
                                key="GaussianMixturePoints.num_centers",
                            ),
                            gamma=Correlated(
                                LogUniform(0.1, 10.0),
                                key="KernelMatrix.scale",
                            ),
                            normalize=False,
                        ),
                    ],
                    correlated=True,
                    key="simple_matrix_type",
                ),
                activation=MixtureActivation(
                    activations=[
                        (fixed_activation, 8.0),
                        ReLUPowActivation(
                            exponent=Correlated(LogUniform(0.1, 10.0)),
                        ),
                        SignPowActivation(
                            exponent=Correlated(LogUniform(0.1, 10.0)),
                        ),
                        InversePowActivation(
                            exponent=Correlated(LogUniform(0.1, 10.0)),
                        ),
                        IntegerPowActivation(
                            exponent=Correlated(UniformInt(2, 5)),
                        ),
                    ],
                    key="matrix_activation_type",
                ),
            ),
        ],
        correlated=True,
        key="matrix_type",
    )

    functions = [
        RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        RandomNN(
            num_layers=Correlated(LogUniformInt(1, 3)),
            hidden_channels=Correlated(LogUniformInt(1, 127)),
            activation=node_activation,
            linear=RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        ),
        RandomTree(
            num_trees=Correlated(LogUniformInt(1, 128)),
            depth=Correlated(UniformInt(1, 7)),
        ),
        RandomDiscretization(
            num_centers=Correlated(LogUniformInt(2, 255)),
            p=Correlated(LogUniform(0.5, 4.0)),
            linear=RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        ),
        RandomQuadratic(matrix_sampler=deepcopy(matrix_sampler)),
        RandomEMAssignment(
            p=Choice(1.0, 2.0),
            q=Correlated(Uniform(1.0, 2.0)),
            bandwidth=Correlated(Uniform(0.0, 1.0)),
            linear=RandomLinear(matrix_sampler=deepcopy(matrix_sampler)),
        ),
        RandomGP(
            a=Correlated(LogUniform(2.0, 20.0)),
            alpha=Correlated(LogUniform(0.5, 10.0)),
            num_features=256,
        ),
    ]
    random_fn = MixtureFunction(
        [
            *functions,
            RandomProduct(MixtureFunction(functions)),
        ]
    )
    parent_merger = MixtureParentMerger(
        [
            ConcatParentMerger(),
            ReduceParentMerger(Choice("sum", "prod", "amax", "logsumexp")),
        ]
    )
    numerical_converter = DistributionalMixtureConverter()
    categorical_converter = MixtureConverter(
        converters=[
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="identity",
            ),
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="index",
            ),
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="center",
            ),
            NearestNeighborConverter(
                p=Correlated(
                    LogUniform(0.5, 4.0), key="NeighborDiscretizer.p"
                ),
                output_mode="function",
                random_fn=random_fn,
            ),
            SoftmaxCategoricalConverter(
                output_mode="identity",
                sample_categories=True,
                argmax_prob=softmax_argmax_prob,
            ),
            SoftmaxCategoricalConverter(
                output_mode="index",
                sample_categories=True,
                argmax_prob=softmax_argmax_prob,
            ),
            SoftmaxCategoricalConverter(
                output_mode="embedding",
                sample_categories=True,
                argmax_prob=softmax_argmax_prob,
                root_sampler=root_sampler,
                random_fn=random_fn,
            ),
        ],
    )
    transforms = (
        pt.Choice(
            pt.CorrelateColumns(),
            pt.Identity(),
            weights=(correlated_block_prob, 1 - correlated_block_prob),
        ),
        pt.ScaleFeatures(),
        pt.ShuffleColumns(),
        pt.Choice(
            pt.Relabel(
                MixtureFunction(
                    [
                        (functions[0], 0.5),
                        (
                            RandomNN(
                                num_layers=Correlated(LogUniformInt(1, 2)),
                                hidden_channels=Correlated(
                                    LogUniformInt(1, 32)
                                ),
                                activation=node_activation,
                                linear=RandomLinear(
                                    matrix_sampler=deepcopy(matrix_sampler)
                                ),
                            ),
                            0.3,
                        ),
                        (functions[-1], 0.2),
                    ]
                )
            ),
            pt.Identity(),
            weights=(relabel_prob, 1 - relabel_prob),
        ),
        pt.Choice(
            pt.Coarsen(categorical=True, standardize=True),
            pt.Identity(),
            weights=(coarsen_prob, 1 - coarsen_prob),
        ),
        pt.Choice(
            pt.RecodeHighCardinality(categorical=True, standardize=True),
            pt.Identity(),
            weights=(high_cardinality_prob, 1 - high_cardinality_prob),
        ),
        pt.Choice(
            pt.WarpTargetTails(),
            pt.ScaleTarget(),
            weights=(y_heavy_tail_prob, 1 - y_heavy_tail_prob),
        )
        if task_type == Task.regression
        else pt.ShuffleClasses(),
        pt.RejectNonfinite(),
        pt.Choice(
            pt.MaskNearEmpty(),
            pt.Choice(
                pt.MaskStructured(),
                pt.Identity(),
                weights=(
                    structured_missingness_prob,
                    1 - structured_missingness_prob,
                ),
            ),
            weights=(
                near_empty_missingness_prob,
                1 - near_empty_missingness_prob,
            ),
        ),
        pt.Choice(
            pt.MaskRandom(Uniform(0.01, 0.08)),
            pt.Identity(),
            weights=(mcar_missingness_prob, 1 - mcar_missingness_prob),
        ),
        pt.Choice(
            pt.MaskLatent(),
            pt.Identity(),
            weights=(
                latent_driver_missingness_prob,
                1 - latent_driver_missingness_prob,
            ),
        ),
    )
    return SCMPrior(
        schema_sampler=schema_sampler,
        dag_sampler=dag_sampler,
        latent_channels=latent_channels,
        root_sampler=root_sampler,
        random_fn=random_fn,
        parent_merger=parent_merger,
        numerical_converter=numerical_converter,
        categorical_converter=categorical_converter,
        transforms=transforms,
    )


import math

import torch
from sdm import Stype
from sdm.prior import should_drop
from sdm.prior.tabular.transform._table import category_values, feature_order

from tfmplayground.configs.priors import KumoClassificationPriorConfig, KumoRegressionPriorConfig
from tfmplayground.priors.base import Prior
from tfmplayground.utils import get_default_device


def _standardize_features(x: torch.Tensor, train_size: int) -> torch.Tensor:
    values = x.double()
    context = values[..., :train_size, :]
    context = context.masked_fill(~context.isfinite(), torch.nan)
    mean = context.nanmean(dim=-2, keepdim=True).nan_to_num(nan=0.0)
    var = (context - mean).square().nanmean(dim=-2, keepdim=True)
    scale = var.nan_to_num(nan=0.0).sqrt() + 1e-6
    return ((values - mean) / scale).clamp(-100.0, 100.0).to(x.dtype)


def _robust_probe_target(y: torch.Tensor) -> torch.Tensor:
    centered = y - y.median()
    mad = 1.4826 * centered.abs().median()
    std = y.std(correction=0)
    scale = torch.where(
        mad > 0, mad, torch.where(std > 0, std, torch.ones_like(std))
    )
    return (centered / scale).clamp(-4.0, 4.0)


class KumoPrior(Prior):
    """
    adapts the kumo tabular prior to prior interface, with the stage 1 settings of its training recipe and no missing
    values
    """

    def __init__(
        self,
        config: KumoClassificationPriorConfig | KumoRegressionPriorConfig,
        device: str | torch.device | None = None,
    ) -> None:
        """
        keeps config, checks config limits and builds the kumo prior
        """
        self.config = config
        self.problem = config.problem
        self.device = device if device is not None else get_default_device()
        if not 0 < config.min_train_fraction <= config.max_train_fraction < 1:
            raise ValueError("train fractions must be 0 < min <= max < 1")
        if not 1 <= config.min_num_features <= config.max_num_features:
            raise ValueError("feature counts must be 1 <= min <= max")
        if not 1 < config.min_num_datapoints <= config.max_num_datapoints:
            raise ValueError("datapoint counts must be 1 < min <= max")
        if self.problem == "classification" and config.max_num_classes < 2:
            raise ValueError(f"classification needs at least 2 classes, not {config.max_num_classes}")
        self.task = Task.classification if self.problem == "classification" else Task.regression
        self.prior = make_prior(
            self.task,
            relabel_prob=0.0,
            coarsen_prob=0.0,
            high_cardinality_prob=0.0,
            y_heavy_tail_prob=0.0,
            structured_missingness_prob=0.0,
            mcar_missingness_prob=0.0,
            near_empty_missingness_prob=0.0,
            latent_driver_missingness_prob=0.0,
            softmax_argmax_prob=0.0,
        )

    def batch_hyperparameters(self) -> None:
        """
        samples sizes every table of the next batch shares
        """
        c = self.config
        self.num_features = int(torch.randint(c.min_num_features, c.max_num_features + 1, ()))
        low, high = c.min_num_datapoints, c.max_num_datapoints
        # log-uniform like their training loader (train/kumo/tabular/loader.py)
        self.num_datapoints = low if low == high else int(torch.empty(()).uniform_(math.log(low), math.log(high)).exp())
        fraction = c.min_train_fraction + float(torch.rand(())) * (c.max_train_fraction - c.min_train_fraction)
        self.sep = min(max(int(self.num_datapoints * fraction), 1), self.num_datapoints - 1)
        classification = self.problem == "classification"
        self.num_classes = int(torch.randint(2, c.max_num_classes + 1, ())) if classification else None

    def dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        samples one table like the kumo training loader: redraws tables the prior rejects, whose train part misses a
        test class, whose features are all constant, or that the extra trees filter finds unlearnable or trivial
        """
        rows, sep = self.num_datapoints, self.sep
        for attempt in range(100):
            schema = self.prior.schema_sampler.sample(num_columns=self.num_features, num_classes=self.num_classes)
            sample = self.prior.sample(num_rows=rows, schema=schema)
            if sample is None:
                continue
            x, y = sample
            if not x.active_stypes <= {Stype.numerical, Stype.categorical}:
                raise ValueError("Training features must be numerical or categorical")
            features = torch.cat((x.numerical, category_values(x.categorical, x.numerical.dtype)), dim=-1)
            features = features[..., feature_order(x)]
            if y.categorical.size(-1):
                target = y.categorical.code.squeeze(-1).long()
                for permutation in range(11):
                    if permutation:
                        order = torch.randperm(rows)
                        candidate = target[order]
                    else:
                        candidate = target
                    context, query = candidate[:sep].unique(), candidate[sep:].unique()
                    if len(context) >= 2 and torch.equal(context, query):
                        if permutation:
                            features, target = features[order], candidate
                        break
                else:
                    continue
                target = target.unique(return_inverse=True)[1]
            else:
                target = y.numerical.squeeze(-1)
                if target.std(correction=0) < 1e-8 or target[:sep].std(correction=0) <= 0.1:
                    continue
            features = features[:, (features[:sep] != features[:1]).any(dim=0)]
            if features.size(-1) == 0:
                continue
            probe = _robust_probe_target(target) if self.task == Task.regression else target
            if attempt < 5 and should_drop(
                x=features, y=probe, task_type=self.task, drop_trivial=True, drop_unlearnable=True
            ):
                continue
            return features, target
        raise RuntimeError(f"No valid dataset after 100 attempts (rows={rows}, split={sep})")

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        stacks sampled tables into one batch, pads features to shared width, standardizes them on train rows and
        splits at train test index
        """
        self.batch_hyperparameters()
        tables = [self.dataset() for _ in range(batch_size)]
        x = torch.zeros(batch_size, self.num_datapoints, self.num_features)
        for i, (features, _) in enumerate(tables):
            x[i, :, : features.size(-1)] = features
        x = _standardize_features(x, self.sep).to(self.device)
        y = torch.stack([target.float() for _, target in tables]).to(self.device)
        return x[:, : self.sep], y[:, : self.sep], x[:, self.sep :], y[:, self.sep :]
