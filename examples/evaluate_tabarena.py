# %% imports
from pathlib import Path

from tfmplayground.evaluation.arena import evaluate_arena

# checkpoint written by examples/pretraining_classification.py
checkpoint_path = sorted(Path("workdir/experiments/classification").glob("**/*-ckpt-last.pth"))[-1]

# official tabarena protocol on tabarena-lite (first split of every dataset), compared against leaderboard
leaderboard = evaluate_arena(
    checkpoint_path,
    arena="tabarena",  # or "beyondarena"
    max_n_samples=10_000,
    results_dir="workdir/tabarena/results",
    output_dir="workdir/tabarena/eval",
)
print(leaderboard.to_markdown())
