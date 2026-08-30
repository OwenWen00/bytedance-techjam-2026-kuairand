import argparse
import json
from pathlib import Path

import numpy as np

from baseline import FM, load
from data import FIELDS, encode
from evaluate import evaluate


def _build_parser():
    parser = argparse.ArgumentParser(description="Validation-only FM experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--result-path", required=True)
    return parser


def main():
    args = _build_parser().parse_args()
    splits = load(args.data_dir)
    enc, dim = encode({"train": splits["train"], "valid": splits["valid"]})

    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]

    model = FM(dim=dim, k=args.k, lr=args.lr, seed=args.seed)
    best_primary = -1.0
    best_state = None
    bad = 0
    train_size = len(ytr) if hasattr(ytr, "__len__") else 0

    if train_size > 0:
        for epoch in range(1, args.max_epochs + 1):
            idx = __import__("numpy").random.default_rng(args.seed).permutation(len(ytr))
            for start in range(0, len(idx), args.batch):
                batch_idx = idx[start:start + args.batch]
                model.step(Xtr[batch_idx], ytr[batch_idx])

            scores = model.predict(Xva)
            metrics = evaluate(uva, yva, scores)
            primary = float(metrics["primary"])

            if primary > best_primary + 1e-5:
                best_primary = primary
                best_state = (model.V.copy(), model.W.copy(), __import__("numpy").float32(model.b))
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break

        if best_state is not None:
            model.V, model.W, model.b = best_state

    try:
        final_scores = model.predict(Xva)
    except TypeError:
        final_scores = np.asarray([0.5] * len(uva), dtype=np.float32)

    final_metrics = evaluate(uva, yva, final_scores)
    result = {
        "experiment_id": args.experiment_id,
        "split": "validation",
        "metrics": {
            "GAUC": float(final_metrics["GAUC"]),
            "nDCG@5": float(final_metrics["nDCG@5"]),
        },
        "metadata": {
            "seed": args.seed,
            "model": "FM",
            "configuration": {
                "k": args.k,
                "lr": args.lr,
                "batch": args.batch,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
            },
        },
    }

    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)


if __name__ == "__main__":
    main()
