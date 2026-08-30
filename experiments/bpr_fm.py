import argparse
import json
from pathlib import Path

import numpy as np

from baseline import FM, load
from data import encode
from evaluate import evaluate
from experiments.sampling import sample_same_user_pairs


def save_checkpoint(model, path: str | Path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "V": np.asarray(model.V, dtype=np.float32),
        "W": np.asarray(model.W, dtype=np.float32),
        "b": np.asarray(model.b, dtype=np.float32),
    }
    if hasattr(model, "t"):
        payload["t"] = np.asarray(int(model.t), dtype=np.int64)
    np.savez(checkpoint_path, **payload)
    return checkpoint_path


def load_checkpoint(path: str | Path, model_cls, **kwargs):
    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as payload:
        model = model_cls(**kwargs)
        model.V = np.asarray(payload["V"], dtype=np.float32).copy()
        model.W = np.asarray(payload["W"], dtype=np.float32).copy()
        model.b = np.asarray(payload["b"], dtype=np.float32).copy()
        if "t" in payload:
            model.t = int(np.asarray(payload["t"]).item())
        return model


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class BPRFM:
    def __init__(self, dim: int, k: int = 16, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0.0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.k = int(k)
        self.dim = int(dim)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def _forward(self, X):
        if X.size == 0:
            raise ValueError("BPR-FM requires at least one row to score.")
        E = self.V[X]
        S = E.sum(axis=1)
        inter = 0.5 * ((S ** 2).sum(axis=1) - (E ** 2).sum(axis=(1, 2)))
        scores = self.b + self.W[X].sum(axis=1) + inter
        return scores.astype(np.float32), E, S

    def predict(self, X, bs: int = 200_000):
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        chunks = [self._forward(X[i:i + bs])[0] for i in range(0, len(X), bs)]
        return np.concatenate(chunks).astype(np.float32)

    def _pair_loss_and_grad(self, X_pos, X_neg):
        pos_scores, pos_E, pos_S = self._forward(X_pos)
        neg_scores, neg_E, neg_S = self._forward(X_neg)
        diff = pos_scores - neg_scores
        sigma = sigmoid(diff)
        loss = -np.mean(np.log(sigma + 1e-9))

        if not np.all(np.isfinite(pos_scores)) or not np.all(np.isfinite(neg_scores)):
            raise ValueError("BPR-FM produced non-finite scores during training.")
        if not np.isfinite(loss):
            raise ValueError("BPR-FM produced a non-finite pairwise loss.")

        grad_factor = sigma - 1.0
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)

        np.add.at(gV, X_pos, grad_factor[:, None, None] * (pos_S[:, None, :] - pos_E))
        np.add.at(gV, X_neg, (-grad_factor)[:, None, None] * (neg_S[:, None, :] - neg_E))
        np.add.at(gW, X_pos, grad_factor[:, None])
        np.add.at(gW, X_neg, (-grad_factor)[:, None])

        gV += self.l2 * self.V
        gW += self.l2 * self.W

        if not np.all(np.isfinite(gV)) or not np.all(np.isfinite(gW)):
            raise ValueError("BPR-FM produced non-finite gradients.")

        return float(loss), gV, gW

    def step(self, X_pos, X_neg):
        if X_pos.shape[0] != X_neg.shape[0]:
            raise ValueError("Positive and negative batches must have the same number of rows.")
        if X_pos.size == 0:
            return 0.0

        loss, gV, gW = self._pair_loss_and_grad(X_pos, X_neg)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1.0 - b1) * G
            Vv *= b2
            Vv += (1.0 - b2) * (G * G)
            P -= self.lr * (M / (1.0 - b1 ** self.t)) / (np.sqrt(Vv / (1.0 - b2 ** self.t)) + eps)

        if not np.all(np.isfinite(self.V)) or not np.all(np.isfinite(self.W)):
            raise ValueError("BPR-FM parameter state became non-finite after an update.")

        return loss


def _build_parser():
    parser = argparse.ArgumentParser(description="Validation-only pairwise BPR-FM experiment")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-pairs-per-user", type=int, default=64)
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
    skipped_users = 0

    for epoch in range(1, args.max_epochs + 1):
        pos_idx, neg_idx, pair_count, skipped = sample_same_user_pairs(
            users_train,
            ytr,
            seed=args.seed,
            epoch=epoch - 1,
            max_pairs_per_user=args.max_pairs_per_user,
        )
        sampled_pairs += pair_count
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
            "model": "BPR-FM",
            "strategy": "pairwise_bpr_fm",
            "checkpoint_path": str(checkpoint_path),
            "configuration": {
                "k": args.k,
                "lr": args.lr,
                "batch": args.batch,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "max_pairs_per_user": args.max_pairs_per_user,
            },
            "sampling": {
                "pairs_sampled": int(sampled_pairs),
                "skipped_users": int(skipped_users),
            },
        },
    }

    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)


if __name__ == "__main__":
    main()
