"""CLI for one validation-only KuaiRand FM experiment.

The autonomous Agent uses this module through a trusted argv adapter.  This
entrypoint intentionally has no test-scoring option.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trial_lab_core import run_experiment


VARIANTS = {
    "pointwise_fm": ("pointwise", "official", "random"),
    "pairwise_bpr": ("pairwise", "official", "random"),
    "hard_negative_bpr": ("pairwise", "official", "hard"),
    "history_pairwise": ("pairwise", "history", "random"),
}

REQUIRED_DATA_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one controlled KuaiRand experiment")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--starter-dir", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-per-positive", type=int, default=1)
    parser.add_argument("--hard-candidates", type=int, default=5)
    parser.add_argument("--hard-negative-warmup", type=int, default=3)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--max-pairs-per-epoch", type=int, default=0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Direct code-connectivity check only; never used by the Agent tool",
    )
    return parser


def _validate_environment(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise ValueError("missing KuaiRand data files: %s" % ", ".join(missing))
    starter_dir = Path(args.starter_dir).expanduser().resolve()
    for name in ("data.py", "evaluate.py"):
        if not (starter_dir / name).is_file():
            raise ValueError("starter directory is missing %s" % name)
    if not 2 <= args.k <= 64:
        raise ValueError("k must be in [2, 64]")
    if not 1e-5 <= args.lr <= 0.1:
        raise ValueError("lr must be in [1e-5, 0.1]")
    if not 0.0 <= args.l2 <= 0.1:
        raise ValueError("l2 must be in [0, 0.1]")
    if not 1 <= args.epochs <= 80 or not 1 <= args.patience <= 10:
        raise ValueError("epochs/patience are outside the controlled range")
    if not 256 <= args.batch_size <= 262144:
        raise ValueError("batch-size is outside the controlled range")
    if not 0 <= args.seed <= 2 ** 31 - 1:
        raise ValueError("seed is outside the controlled range")
    if not 1 <= args.negative_per_positive <= 5:
        raise ValueError("negative-per-positive must be in [1, 5]")
    if not 2 <= args.hard_candidates <= 20:
        raise ValueError("hard-candidates must be in [2, 20]")
    if not 0 <= args.hard_negative_warmup <= 10:
        raise ValueError("hard-negative-warmup must be in [0, 10]")
    if not 0.0 <= args.hard_negative_ratio <= 1.0:
        raise ValueError("hard-negative-ratio must be in [0, 1]")
    if not 0 <= args.max_pairs_per_epoch <= 2_000_000:
        raise ValueError("max-pairs-per-epoch is outside the controlled range")


def main() -> None:
    args = build_parser().parse_args()
    _validate_environment(args)
    args.score_test = False
    args.max_train_rows = 50_000 if args.smoke else 0
    args.max_eval_rows = 20_000 if args.smoke else 0
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_pairs_per_epoch = 20_000
    training_mode, encoder_mode, negative_strategy = VARIANTS[args.variant]
    summary = run_experiment(
        args,
        training_mode=training_mode,
        encoder_mode=encoder_mode,
        negative_strategy=negative_strategy,
    )
    if summary.get("test") is not None:
        raise RuntimeError("validation-only runner produced forbidden test metrics")
    print(json.dumps(summary["valid"], sort_keys=True))


if __name__ == "__main__":
    main()
