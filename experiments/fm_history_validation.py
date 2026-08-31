import argparse
import json
from pathlib import Path

import numpy as np

from baseline import FM
from data import load
from evaluate import evaluate
from experiments.history_features import append_history_feature, build_causal_history_features


def _build_parser():
    parser = argparse.ArgumentParser(description="Validation-only FM experiment with train-only causal history feature")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    return parser


def _encode_with_history(splits):
    from data import encode

    enc, base_dim = encode({"train": splits["train"], "valid": splits["valid"]})
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    train_hist, valid_hist, _ = build_causal_history_features(splits["train"], splits["valid"])
    Xtr_aug = append_history_feature(Xtr, train_hist, bucket_count=len(np.linspace(0.0, 1.0, 11)) - 1)
    Xva_aug = append_history_feature(Xva, valid_hist, bucket_count=len(np.linspace(0.0, 1.0, 11)) - 1)
    return Xtr_aug, ytr, Xva_aug, yva, uva, base_dim + (len(np.linspace(0.0, 1.0, 11)) - 1)


def main():
    args = _build_parser().parse_args()
    splits = load(args.data_dir)
    Xtr, ytr, Xva, yva, uva, dim = _encode_with_history(splits)

    model = FM(dim=dim, k=args.k, lr=args.lr, seed=args.seed)
    best_primary = -1.0
    best_state = None
    bad = 0

    rng = np.random.default_rng(args.seed)
    track = range(1, args.max_epochs + 1)
    for epoch in track:
        idx = rng.permutation(len(ytr))
        for start in range(0, len(idx), args.batch):
            batch_idx = idx[start:start + args.batch]
            model.step(Xtr[batch_idx], ytr[batch_idx])

        scores = model.predict(Xva)
        metrics = evaluate(uva, yva, scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-5:
            best_primary = primary
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.V, model.W, model.b = best_state

    final_scores = model.predict(Xva)
    final_metrics = evaluate(uva, yva, final_scores)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("artifacts/checkpoints") / f"{args.experiment_id}.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "V": np.asarray(model.V, dtype=np.float32),
        "W": np.asarray(model.W, dtype=np.float32),
        "b": np.asarray(model.b, dtype=np.float32),
    }
    np.savez(checkpoint_path, **payload)

    result = {
        "experiment_id": args.experiment_id,
        "split": "validation",
        "metrics": {
            "GAUC": float(final_metrics["GAUC"]),
            "nDCG@5": float(final_metrics["nDCG@5"]),
        },
        "metadata": {
            "seed": args.seed,
            "model": "history_fm",
            "checkpoint_path": str(checkpoint_path),
            "configuration": {
                "k": args.k,
                "lr": args.lr,
                "batch": args.batch,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "history_feature": "causal_user_author_long_view_rate",
                "history_buckets": 10,
            },
        },
    }
    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)


if __name__ == "__main__":
    main()
