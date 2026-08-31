import argparse
import json
from pathlib import Path

import numpy as np

from baseline import FM, load
from data import encode
from evaluate import evaluate
from experiments.bpr_fm import BPRFM, save_checkpoint, load_checkpoint
from experiments.hard_negative_sampling import sample_hard_negative_pairs


def _build_parser():
    parser = argparse.ArgumentParser(description="Hard-negative BPR-FM validation experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-pairs-per-user", type=int, default=64)
    parser.add_argument("--hard-negative-candidates", type=int, default=64)
    parser.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    return parser


def main():
    args = _build_parser().parse_args()
    splits = load(args.data_dir)
    enc, dim = encode({"train": splits["train"], "valid": splits["valid"]})

    Xtr, ytr, users_train = enc["train"]
    Xva, yva, uva = enc["valid"]

    model = BPRFM(dim=dim, k=args.k, lr=args.lr, seed=args.seed)
    best_primary = -1.0
    best_state = None
    bad = 0
    sampled_pairs = 0
    total_candidates = 0
    skipped_users = 0
    epochs_completed = 0

    for epoch in range(1, args.max_epochs + 1):
        epochs_completed = epoch
        model_scores = model.predict(Xtr)
        pos_idx, neg_idx, pair_count, candidate_count, skipped = sample_hard_negative_pairs(
            users_train,
            ytr,
            model_scores,
            seed=args.seed,
            epoch=epoch - 1,
            max_pairs_per_user=args.max_pairs_per_user,
            hard_negative_candidates=args.hard_negative_candidates,
        )
        sampled_pairs += pair_count
        total_candidates += candidate_count
        skipped_users += skipped

        if pair_count == 0:
            continue

        for start in range(0, pair_count, args.batch):
            batch_pos = pos_idx[start:start + args.batch]
            batch_neg = neg_idx[start:start + args.batch]
            if batch_pos.size == 0 or batch_neg.size == 0:
                continue
            model.step(Xtr[batch_pos], Xtr[batch_neg])

        scores = model.predict(Xva)
        metrics = evaluate(uva, yva, scores)
        primary = float(metrics["primary"])

        if primary > best_primary + 1e-5:
            best_primary = primary
            best_state = (model.V.copy(), model.W.copy())
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.V, model.W = best_state

    final_scores = model.predict(Xva)
    final_metrics = evaluate(uva, yva, final_scores)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("artifacts/checkpoints") / f"{args.experiment_id}.npz"
    checkpoint_path = save_checkpoint(model, checkpoint_path)

    result = {
        "experiment_id": args.experiment_id,
        "split": "validation",
        "metrics": {
            "GAUC": float(final_metrics["GAUC"]),
            "nDCG@5": float(final_metrics["nDCG@5"]),
        },
        "metadata": {
            "seed": args.seed,
            "model": "Hard-Negative BPR-FM",
            "strategy": "hard_negative_bpr_fm",
            "checkpoint_path": str(checkpoint_path),
            "configuration": {
                "k": args.k,
                "lr": args.lr,
                "batch": args.batch,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "max_pairs_per_user": args.max_pairs_per_user,
                "hard_negative_candidates": args.hard_negative_candidates,
            },
            "sampling": {
                "counter_scope": "cumulative_across_epochs",
                "epochs_completed": int(epochs_completed),
                "sampled_pair_count": int(sampled_pairs),
                "candidate_count": int(total_candidates),
                "skipped_user_count": int(skipped_users),
                "skipped_user_count_unit": "user_epochs",
                "sampler_name": "hard_negative",
            },
        },
    }

    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)


if __name__ == "__main__":
    main()
