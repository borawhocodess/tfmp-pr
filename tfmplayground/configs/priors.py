from dataclasses import dataclass


@dataclass
class NanoTabICLPriorConfig:
    """
    settings nanotabicl priors share
    """

    min_num_datapoints: int = 1000
    max_num_datapoints: int = 1000
    min_num_features: int = 2
    max_num_features: int = 20
    min_train_fraction: float = 0.1
    max_train_fraction: float = 0.9
    max_cat_size: int = 100


@dataclass
class NanoTabICLClassificationPriorConfig(NanoTabICLPriorConfig):
    """
    nanotabicl prior settings for classification
    """

    problem: str = "classification"
    max_num_classes: int = 10
    binary_class_probability: float = 0.5
    max_row_permutations: int = 11


@dataclass
class NanoTabICLRegressionPriorConfig(NanoTabICLPriorConfig):
    """
    nanotabicl prior settings for regression
    """

    problem: str = "regression"
    max_num_classes: int = 0


@dataclass
class SCMPriorConfig:
    """
    settings scm priors share, defaults follow the tabpfnv2 ranges where a split leaves enough train rows
    """

    min_num_datapoints: int = 20
    max_num_datapoints: int = 2048
    min_num_features: int = 1
    max_num_features: int = 160
    min_train_fraction: float = 0.1
    max_train_fraction: float = 0.9


@dataclass
class SCMClassificationPriorConfig(SCMPriorConfig):
    """
    scm prior settings for classification
    """

    problem: str = "classification"
    min_num_datapoints: int = 100
    max_num_classes: int = 10


@dataclass
class SCMRegressionPriorConfig(SCMPriorConfig):
    """
    scm prior settings for regression
    """

    problem: str = "regression"


@dataclass
class TabICLPriorConfig:
    """
    settings tabicl priors share
    """

    num_datapoints_min: int = 128
    num_datapoints_max: int = 1024
    num_features_min: int = 2
    num_features_max: int = 100
    prior_type: str = "graph_scm"
    n_jobs: int = 1
    filter_unpredictable_datasets: bool = True
    filter_unpredictable_graphs: bool = True


@dataclass
class TabICLClassificationPriorConfig(TabICLPriorConfig):
    """
    tabicl prior settings for classification
    """

    problem: str = "classification"
    max_num_classes: int = 10


@dataclass
class TabICLRegressionPriorConfig(TabICLPriorConfig):
    """
    tabicl prior settings for regression
    """

    problem: str = "regression"
    max_num_classes: int = 0


@dataclass
class PriorDumpConfig:
    """
    settings prior dumps share
    """

    filename: str = ""
    starting_index: int = 0


@dataclass
class ClassificationPriorDumpConfig(PriorDumpConfig):
    """
    prior dump settings for classification
    """

    problem: str = "classification"
    filename: str = "50x3_3_100k_classification.h5"


@dataclass
class RegressionPriorDumpConfig(PriorDumpConfig):
    """
    prior dump settings for regression
    """

    problem: str = "regression"
    filename: str = "50x3_1280k_regression.h5"
