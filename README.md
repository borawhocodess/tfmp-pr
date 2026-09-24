# TFM-Playground

The purpose of this repository is to provide a fully open source playground for tabular foundation models.

```
todo: clone and uv instructions (to set it up as package editable?)
```

### Pretrain a TFM with configurations

```python
modelconfig = ...
priorconfig = ...
evalconfig = ...
trainconfig = ...
experimentconfig=...

model = pretrainTFM(
    problem="classification",
    model=...Model(config=modelconfig),
    prior=...Prior(config=priorconfig, device=device),
    eval=evalconfig,
    training=trainconfig,
    experiment=experimentconfig,
)
```

full examples in [examples/pretraining_classification.py](examples/pretraining_classification.py) and  [examples/pretraining_regression.py](examples/pretraining_regression.py)

### model list

- nanotabpfn - [adapter](tfmplayground/models/nanotabpfn.py) · [repo](https://github.com/automl/nanoTabPFN) · [paper](https://arxiv.org/abs/2511.03634)
- moddednanotabpfn - [adapter](tfmplayground/models/moddednanotabpfn.py) · [repo](https://github.com/borawhocodess/modded-nanotabpfn) · [paper](https://arxiv.org/abs/2606.03681)
- nanotabicl - [adapter](tfmplayground/models/nanotabicl.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/models/tabicl.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)
- tabfm - [adapter](tfmplayground/models/tabfm.py) · [repo](https://github.com/google-research/tabfm) · [blog](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)

### prior list 

- nanotabicl - [adapter](tfmplayground/priors/nanotabicl.py) · [repo](https://github.com/soda-inria/nanotabicl)
- tabicl - [adapter](tfmplayground/priors/tabicl.py) · [repo](https://github.com/soda-inria/tabicl) · [paper](https://arxiv.org/abs/2602.11139)

### eval stuff 

periodic evaluation during pretraining uses [EvaluationConfig](tfmplayground/configs/evaluation.py) on openml tasks as a fast proxy.

for the official score, pretrained checkpoints run through [TabArena](https://github.com/autogluon/tabarena)'s own evaluation: its tasks and splits, preprocessing, validation protocol (bagging), metrics and leaderboard, for both TabArena and BeyondArena. [adapter](tfmplayground/evaluation/arena.py)

```python
from tfmplayground.evaluation.arena import evaluate_arena

leaderboard = evaluate_arena("path/to/ckpt-best.pth", arena="tabarena")  # or "beyondarena"
```

or from the command line

```
python -m tfmplayground.evaluation.arena path/to/ckpt-best.pth --arena tabarena --output_dir workdir/tabarena/eval
```

defaults are tabarena-lite (beyondarena core) tasks of checkpoint problem, `--outer` skips bagging for a faster unofficial run, and `--ray` runs jobs in parallel. full example in [examples/evaluate_tabarena.py](examples/evaluate_tabarena.py)

### train stuff 
