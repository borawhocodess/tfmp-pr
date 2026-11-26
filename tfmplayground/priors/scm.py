import numpy as np
import torch

from tfmplayground.configs.priors import SCMClassificationPriorConfig, SCMRegressionPriorConfig
from tfmplayground.priors.base import Prior
from tfmplayground.utils import get_default_device

ACTIVATIONS = [
    lambda x: x,
    lambda x: torch.log(torch.clamp(torch.abs(x), min=1e-6)),
    torch.sigmoid,
    torch.abs,
    torch.sin,
    torch.tanh,
    lambda x: torch.argsort(torch.argsort(x, dim=-1), dim=-1).float(),
    torch.square,
    lambda x: torch.sign(x) * torch.abs(x) ** float(np.exp(np.random.uniform(np.log(0.1), np.log(10)))),
    torch.nn.functional.softplus,
    lambda x: (x > 0).float(),
    lambda x: x - torch.floor(x),
]


def table_hyperparameters(num_features: int) -> dict:
    """
    samples the causal structure of one table

    from the tabpfnv2 paper: node count, redirection, subgraphs, root mechanism, prototype mixing and edge dimension;
    ranges the paper does not give are taken from tabicl (scale), tabpfnv1 (noise) and nanotabicl (categorical share)
    """
    nodes = int(round(np.exp(np.random.uniform(np.log(4), np.log(32)))))
    dimension = int(round(np.exp(np.random.uniform(np.log(2), np.log(16)))))
    return {
        "nodes": nodes,
        "redirection": min(np.random.gamma(4.0, 1 / 8), 1.0),
        "subgraphs": int(min(np.random.geometric(0.7), 4)),
        "mechanism": str(np.random.choice(["normal", "uniform", "mixed"])),
        "scale": float(np.exp(np.random.uniform(np.log(0.01), np.log(10)))),
        "sigma": float(np.exp(np.random.uniform(np.log(1e-4), np.log(0.3)))),
        "prototypes": float(np.random.uniform(0.1, 0.9)) if np.random.rand() < 0.5 else 0.0,
        "temperature": float(np.exp(np.random.uniform(np.log(0.05), np.log(5.0)))),
        "dimension": max(dimension, -(-(num_features + 1) // nodes)),
        "categoricals": float(np.clip(np.random.uniform(-0.5, 1.2), 0.0, 1.0)),
    }


def growing_network(nodes: int, redirection: float, attempts: int = 3) -> list[list[int]]:
    """
    gives parents of every node in a growing network with redirection, tabpfnv2 graph structure sampling
    """
    parents = [[] for _ in range(nodes)]
    for child in range(1, nodes):
        chosen = set()
        for _ in range(attempts):
            target = int(np.random.randint(child))
            if np.random.rand() < redirection and parents[target]:
                target = int(np.random.choice(parents[target]))
            chosen.add(target)
        parents[child] = sorted(chosen)
    return parents


def graph(nodes: int, redirection: float, subgraphs: int) -> list[list[int]]:
    """
    gives parents of every node in disjoint growing networks, tabpfnv2 graph structure sampling
    """
    parents = []
    for part in np.array_split(np.arange(nodes), subgraphs):
        offset = len(parents)
        parents.extend([[parent + offset for parent in ps] for ps in growing_network(len(part), redirection)])
    return parents


def neural(x: torch.Tensor) -> torch.Tensor:
    """
    random linear map and activation, tabpfnv2 computational edge mappings
    """
    weight = torch.empty(x.shape[-1], x.shape[-1])
    torch.nn.init.xavier_uniform_(weight)
    return ACTIVATIONS[np.random.randint(len(ACTIVATIONS))](x @ weight.T + torch.randn(x.shape[-1]))


def categories(x: torch.Tensor, max_categories: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    gives category of every row by its nearest prototype row, and a random embedding per category, tabpfnv2
    computational edge mappings
    """
    count = int(round(np.random.gamma(2.0, 1.0))) + 2
    count = min(count, len(x), max_categories or count)
    prototypes = x[torch.randperm(len(x))[:count]]
    classes = torch.cdist(x, prototypes).argmin(dim=-1)
    return classes, torch.randn(count, x.shape[-1])[classes]


def tree(x: torch.Tensor) -> torch.Tensor:
    """
    random decision tree with a random value per leaf, tabpfnv2 computational edge mappings
    """
    depth = int(np.random.randint(1, 5))
    dims = torch.randint(x.shape[-1], (depth,))
    thresholds = x[torch.randint(len(x), (depth,)), dims]
    leaves = ((x[:, dims] > thresholds).long() * (2 ** torch.arange(depth))).sum(-1)
    return torch.randn(2**depth, x.shape[-1])[leaves]


def roots(n: int, h: dict) -> torch.Tensor:
    """
    samples values of a node without parents, optionally mixed from prototype rows, tabpfnv2 initialization data
    sampling
    """
    mechanism = h["mechanism"] if h["mechanism"] != "mixed" else str(np.random.choice(["normal", "uniform"]))
    x = h["scale"] * (
        torch.randn(n, h["dimension"]) if mechanism == "normal" else 2 * torch.rand(n, h["dimension"]) - 1
    )
    if h["prototypes"] == 0.0:
        return x
    m = max(int(round(h["prototypes"] * n)), 1)
    weights = torch.from_numpy(np.random.dirichlet(np.full(m, h["temperature"]), size=n)).float()
    return weights @ x[torch.randperm(n)[:m]]


def propagate(parents: list[list[int]], n: int, h: dict) -> list[torch.Tensor]:
    """
    computes values of every node from its parents through neural, tree or categorical edges with noise, tabpfnv2
    causal generative process
    """
    values = []
    for ps in parents:
        if not ps:
            values.append(roots(n, h))
            continue
        edges = []
        for p in ps:
            module = str(np.random.choice(["neural", "tree", "categorical"], p=[0.7, 0.15, 0.15]))
            if module == "neural":
                out = neural(values[p])
            elif module == "tree":
                out = tree(values[p])
            else:
                out = categories(values[p])[1]
            edges.append(out + abs(float(np.random.normal(0.0, h["sigma"]))) * torch.randn_like(out))
        value = torch.stack(edges).mean(0)
        values.append((value - value.mean(0)) / (value.std(0) + 1e-6))
    return values


def warp(x: torch.Tensor) -> torch.Tensor:
    """
    kumaraswamy warping of a column scaled to [0, 1], tabpfnv2 post-processing
    """
    a, b = (float(np.exp(np.random.uniform(np.log(0.2), np.log(5.0)))) for _ in range(2))
    low, high = x.min(dim=0).values, x.max(dim=0).values
    x = torch.clamp((x - low) / (high - low + 1e-30), 0.0, 1.0)
    return 1.0 - (1.0 - x**a) ** b


def quantize(x: torch.Tensor) -> torch.Tensor:
    """
    bins a column at random row values, tabpfnv2 post-processing
    """
    buckets = int(round(np.random.gamma(2.0, 1.0))) + 2
    edges = x[torch.randint(len(x), (buckets + 1,)), 0].sort().values
    return torch.bucketize(x[:, 0].contiguous(), edges[1:-1]).float().unsqueeze(-1)


class SCMPrior(Prior):
    """
    structural causal model prior after the tabpfnv2 paper, written from scratch since tabpfnv2 did not release its
    prior
    """

    def __init__(
        self,
        config: SCMClassificationPriorConfig | SCMRegressionPriorConfig,
        device: str | torch.device | None = None,
    ) -> None:
        """
        keeps config, and checks config limits
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
        min_num_train_rows = int(config.min_num_datapoints * config.min_train_fraction)
        min_num_test_rows = config.min_num_datapoints - int(config.min_num_datapoints * config.max_train_fraction)
        if min_num_train_rows < 2:
            raise ValueError(f"train part needs at least 2 rows, holds {min_num_train_rows} rows")
        if min_num_test_rows < 1:
            raise ValueError(f"test part needs at least 1 row, holds {min_num_test_rows} rows")
        if self.problem == "classification":
            if config.max_num_classes < 2:
                raise ValueError(f"classification needs at least 2 classes, not {config.max_num_classes}")
            if min_num_train_rows < config.max_num_classes:
                raise ValueError(f"{min_num_train_rows} train rows cannot hold {config.max_num_classes} classes")

    def batch_hyperparameters(self) -> None:
        """
        samples sizes every table of the next batch shares, features and the 75000 cell cap after tabpfnv2
        """
        c = self.config
        features = int(round(1 + np.random.beta(0.95, 8.0) * 159))
        self.num_features = int(np.clip(features, c.min_num_features, c.max_num_features))
        rows = int(np.random.randint(c.min_num_datapoints, c.max_num_datapoints + 1))
        self.num_datapoints = max(c.min_num_datapoints, min(rows, 75000 // self.num_features))
        self.sep = int(self.num_datapoints * np.random.uniform(c.min_train_fraction, c.max_train_fraction))

    def target(self, values: list[torch.Tensor]) -> torch.Tensor:
        """
        makes target from a random node, its categories for classification or one of its columns for regression
        """
        node = values[np.random.randint(len(values))]
        if self.problem == "regression":
            return node[:, np.random.randint(node.shape[-1])]
        classes, _ = categories(node, self.config.max_num_classes)
        return classes.unique(return_inverse=True)[1].float()

    def features(self, values: list[torch.Tensor], h: dict) -> torch.Tensor:
        """
        observes features from random node columns, some turned categorical, then warps or bins some continuous ones
        """
        slots = [(node, column) for node in range(len(values)) for column in range(values[node].shape[-1])]
        order = np.random.permutation(len(slots))
        warping, binning = np.random.rand() < 0.5, np.random.rand() < 0.5
        columns = []
        for i in range(self.num_features):
            if np.random.rand() < h["categoricals"]:
                columns.append(categories(values[np.random.randint(len(values))])[0].float().unsqueeze(-1))
                continue
            node, column = slots[order[i % len(slots)]]
            x = values[node][:, column : column + 1]
            if warping and np.random.rand() < 0.3:
                x = warp(x)
            elif binning and np.random.rand() < 0.3:
                x = quantize(x)
            columns.append(x)
        return torch.cat(columns, dim=-1)

    def order(self, y: torch.Tensor) -> torch.Tensor:
        """
        gives row order that puts one row of every class into the train part, so no table is thrown away
        """
        perm = torch.randperm(len(y))
        if self.problem == "regression":
            return perm
        y = y[perm]
        first = torch.stack([(y == c).nonzero()[0, 0] for c in y.unique()])
        rest = torch.ones(len(y), dtype=torch.bool)
        rest[first] = False
        order = torch.cat([first, rest.nonzero()[:, 0]])
        train = order[: self.sep][torch.randperm(self.sep)]
        return perm[torch.cat([train, order[self.sep :]])]

    def dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        samples one table from a fresh causal structure
        """
        h = table_hyperparameters(self.num_features)
        values = propagate(graph(h["nodes"], h["redirection"], h["subgraphs"]), self.num_datapoints, h)
        x, y = self.features(values, h), self.target(values)
        rows = self.order(y)
        return x[rows], y[rows]

    def batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        stacks sampled tables into one batch, split at train test index
        """
        self.batch_hyperparameters()
        datasets = [self.dataset() for _ in range(batch_size)]
        x = torch.stack([d[0] for d in datasets]).to(self.device)
        y = torch.stack([d[1] for d in datasets]).to(self.device)
        return x[:, : self.sep], y[:, : self.sep], x[:, self.sep :], y[:, self.sep :]
