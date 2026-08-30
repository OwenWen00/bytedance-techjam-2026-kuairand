import argparse
import json
import time
from pathlib import Path

import numpy as np


def _state_path(checkpoint_path: str | None, experiment_id: str) -> Path:
    base_dir = Path(checkpoint_path).parent if checkpoint_path else Path("artifacts/checkpoints")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{experiment_id}.state.json"


def _load_attempt_count(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    count = payload.get("attempts", 0)
    if isinstance(count, bool) or not isinstance(count, int):
        return 0
    return max(0, count)


def _write_attempt_count(state_path: Path, attempts: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"attempts": int(attempts)}, sort_keys=True), encoding="utf-8")


def _build_parser():
    parser = argparse.ArgumentParser(description="Synthetic recovery demo")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="validation")
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--ga", type=float, default=None)
    parser.add_argument("--ndcg", type=float, default=None)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--failure-mode", default="none")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.failure_mode not in {"none", "timeout", "error"}:
        raise ValueError(f"Unsupported recovery_demo failure_mode: {args.failure_mode!r}")
    if args.sleep_sec > 0:
        time.sleep(args.sleep_sec)

    state_path = _state_path(args.checkpoint_path, args.experiment_id)
    attempts = _load_attempt_count(state_path)
    if args.failure_mode in {"timeout", "error"} and attempts == 0:
        _write_attempt_count(state_path, 1)
        raise RuntimeError(f"Synthetic {args.failure_mode} triggered for {args.experiment_id}.")

    _write_attempt_count(state_path, attempts + 1)
    if args.ga is not None:
        ga = float(args.ga)
    else:
        base = 0.61 if args.failure_mode == "none" else 0.59
        ga = float(base)
    if args.ndcg is not None:
        ndcg = float(args.ndcg)
    else:
        ndcg = ga + 0.01
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("artifacts/checkpoints") / f"{args.experiment_id}.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_path,
        V=np.zeros((2, 2), dtype=np.float32),
        W=np.zeros(2, dtype=np.float32),
        b=np.zeros(1, dtype=np.float32),
    )

    result = {
        "experiment_id": args.experiment_id,
        "split": "validation",
        "metrics": {"GAUC": ga, "nDCG@5": ndcg},
        "metadata": {
            "seed": args.seed,
            "model": "recovery_demo",
            "strategy": "recovery_demo",
            "mode": args.mode,
            "checkpoint_path": str(checkpoint_path),
            "configuration": {"failure_mode": args.failure_mode, "sleep_sec": args.sleep_sec, "mode": args.mode},
        },
    }
    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)


if __name__ == "__main__":
    main()
