"""Shared experiment engine for the KuaiRand-Pure trial lab.

The official starter-kit files are imported read-only.  This module adds
checkpointing, JSON logs, validation-only model selection, pairwise BPR
training, hard-negative sampling, and a leakage-safe history encoder.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np


OFFICIAL_VALID_PRIMARY = 0.6016


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_modules(starter_dir: str):
    root = Path(starter_dir).expanduser().resolve()
    data_mod = load_module(root / "data.py", "kuairand_official_data")
    eval_mod = load_module(root / "evaluate.py", "kuairand_official_evaluate")
    return data_mod, eval_mod


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """FM compatible with the official implementation plus BPR training."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _apply(self, gV, gW, gb):
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (
                np.sqrt(Vv / (1 - b2 ** self.t)) + eps
            )
        self.b -= self.lr * np.float32(gb)

    def step_pointwise(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        p = sigmoid(z)
        g = ((p - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        self._apply(gV, gW, g.sum())
        return float(
            -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        )

    def step_pairwise(self, X_pos, X_neg):
        """BPR: teach the FM that score(positive) > score(negative)."""
        B = len(X_pos)
        zp, Ep, Sp = self.logits(X_pos)
        zn, En, Sn = self.logits(X_neg)
        delta = zp - zn
        # d[-log(sigmoid(delta))] / d delta
        g = (-sigmoid(-delta) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])
        np.add.at(gV, X_pos, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, X_neg, -g[:, None, None] * (Sn[:, None, :] - En))
        # The global bias cancels in score(pos)-score(neg), so gb is zero.
        self._apply(gV, gW, 0.0)
        return float(np.logaddexp(0.0, -delta).mean())

    def predict(self, X, bs=200_000):
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [self.logits(X[i : i + bs])[0] for i in range(0, len(X), bs)]
        )

    def state(self):
        return {
            "V": self.V.copy(),
            "W": self.W.copy(),
            "b": np.float32(self.b),
            "mV": self.mV.copy(),
            "vV": self.vV.copy(),
            "mW": self.mW.copy(),
            "vW": self.vW.copy(),
            "t": np.int64(self.t),
        }

    def restore(self, state):
        for key, value in state.items():
            setattr(self, key, value.copy() if hasattr(value, "copy") else value)

    def save(self, path: Path):
        np.savez_compressed(path, **self.state())


def make_parser(description: str, default_output: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="KuaiRand-Pure data directory",
    )
    parser.add_argument(
        "--starter-dir",
        default=".",
        help="Official starter-kit directory containing data.py/evaluate.py",
    )
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-per-positive", type=int, default=1)
    parser.add_argument("--hard-candidates", type=int, default=5)
    parser.add_argument(
        "--hard-negative-warmup",
        type=int,
        default=3,
        help="Use random negatives for this many epochs before hard mining",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.5,
        help="After warmup, fraction of pairs using a mined hard negative",
    )
    parser.add_argument(
        "--max-pairs-per-epoch",
        type=int,
        default=0,
        help="0 means all sampled pairs; use a small value for a quick smoke test",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="0 means full train; only use a limit for code smoke tests",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=0,
        help="0 means full validation; only use a limit for code smoke tests",
    )
    parser.add_argument(
        "--score-test",
        action="store_true",
        help="Evaluate test once after training. Do not expose this to an iterative agent.",
    )
    return parser


def append_jsonl(path: Path, record):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record):
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)


def write_predictions(path: Path, rows, scores):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (row, score) in enumerate(zip(rows, scores)):
            writer.writerow([i, row[1], row[2], f"{float(score):.8g}"])


def validate_predictions(path: Path, rows):
    """Apply the official submission alignment and finiteness invariants."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != ["row_id", "user_id", "video_id", "score"]:
            raise ValueError("invalid prediction header")
        count = 0
        for count, record in enumerate(reader, start=1):
            if len(record) != 4:
                raise ValueError("prediction row must contain four columns")
            row_id, user_id, video_id, score_text = record
            expected_index = count - 1
            if expected_index >= len(rows):
                raise ValueError("prediction contains more rows than validation")
            if int(row_id) != expected_index:
                raise ValueError("prediction row_id is not contiguous")
            if user_id != rows[expected_index][1] or video_id != rows[expected_index][2]:
                raise ValueError("prediction row is not aligned with validation data")
            score = float(score_text)
            if not math.isfinite(score):
                raise ValueError("prediction score must be finite")
    if count != len(rows):
        raise ValueError("prediction row count does not match validation")


def encode_official(data_dir: str, official_data):
    splits = official_data.load(data_dir)
    encoded, dim = official_data.encode(splits)
    return splits, encoded, dim, list(official_data.FIELDS)


def _count_bucket(count):
    return str(min(15, int(math.log2(count + 1))))


def _rate_bucket(pos, count):
    if count == 0:
        return "UNK"
    return str(min(10, int(10.0 * pos / count)))


def encode_history(data_dir: str):
    """Build time-safe history features.

    Train rows use statistics from strictly earlier dates.  Validation and test
    rows use only the completed training window.  No validation/test behavior is
    fed back into features.
    """
    data_dir = Path(data_dir)
    video_to_author = {}
    with (data_dir / "video_features_basic_pure.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            video_to_author[row["video_id"]] = row["author_id"]

    split_ranges = {
        "train": (20220408, 20220421),
        "valid": (20220422, 20220428),
        "test": (20220429, 20220508),
    }
    splits = {name: [] for name in split_ranges}
    hour_values = {name: [] for name in split_ranges}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (data_dir / filename).open(newline="") as fh:
            for row in csv.DictReader(fh):
                date = int(row["date"])
                target = None
                for name, (lo, hi) in split_ranges.items():
                    if lo <= date <= hi:
                        target = name
                        break
                if target is None:
                    continue
                user = row["user_id"]
                video = row["video_id"]
                y = 0 if row["long_view"] == "0" else 1
                raw = (
                    date,
                    user,
                    video,
                    video_to_author.get(video, "UNK"),
                    row["tab"],
                    float(row["duration_ms"]),
                    y,
                )
                splits[target].append(raw)
                hour_values[target].append(int(row["hourmin"]) // 100)

    train = splits["train"]
    by_date = collections.defaultdict(list)
    for index, row in enumerate(train):
        by_date[row[0]].append(index)

    user_count = collections.Counter()
    user_pos = collections.Counter()
    item_count = collections.Counter()
    item_pos = collections.Counter()
    train_history = [None] * len(train)
    for date in sorted(by_date):
        indices = by_date[date]
        # Read history before this date.
        for index in indices:
            row = train[index]
            user, video = row[1], row[2]
            train_history[index] = (
                _count_bucket(user_count[user]),
                _rate_bucket(user_pos[user], user_count[user]),
                _count_bucket(item_count[video]),
                _rate_bucket(item_pos[video], item_count[video]),
            )
        # Update only after all rows on the date have received their features.
        for index in indices:
            row = train[index]
            user, video, y = row[1], row[2], row[6]
            user_count[user] += 1
            user_pos[user] += y
            item_count[video] += 1
            item_pos[video] += y

    frozen_history = {}
    for name in ("valid", "test"):
        frozen_history[name] = [
            (
                _count_bucket(user_count[row[1]]),
                _rate_bucket(user_pos[row[1]], user_count[row[1]]),
                _count_bucket(item_count[row[2]]),
                _rate_bucket(item_pos[row[2]], item_count[row[2]]),
            )
            for row in splits[name]
        ]

    duration_edges = np.quantile(
        np.asarray([row[5] for row in train]), np.linspace(0, 1, 11)[1:-1]
    )
    feature_names = [
        "user_id",
        "video_id",
        "author_id",
        "tab",
        "dur_bucket",
        "hour_bucket",
        "weekday",
        "user_past_count_bucket",
        "user_past_long_view_rate_bucket",
        "item_past_count_bucket",
        "item_past_long_view_rate_bucket",
    ]

    def tokens(row, hour, hist):
        date_obj = dt.datetime.strptime(str(row[0]), "%Y%m%d")
        return [
            row[1],
            row[2],
            row[3],
            row[4],
            str(int(np.searchsorted(duration_edges, row[5]))),
            str(hour),
            str(date_obj.weekday()),
            *hist,
        ]

    train_tokens = [
        tokens(row, hour_values["train"][i], train_history[i])
        for i, row in enumerate(train)
    ]
    vocabularies = [dict() for _ in feature_names]
    for values in train_tokens:
        for index, value in enumerate(values):
            if value not in vocabularies[index]:
                vocabularies[index][value] = len(vocabularies[index])
    unknown = [len(v) for v in vocabularies]
    field_dims = [len(v) + 1 for v in vocabularies]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    encoded = {}
    for name in ("train", "valid", "test"):
        rows = splits[name]
        histories = train_history if name == "train" else frozen_history[name]
        token_rows = (
            train_tokens
            if name == "train"
            else [
                tokens(row, hour_values[name][i], histories[i])
                for i, row in enumerate(rows)
            ]
        )
        X = np.empty((len(rows), len(feature_names)), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        for row_index, (row, values) in enumerate(zip(rows, token_rows)):
            for field_index, value in enumerate(values):
                X[row_index, field_index] = (
                    vocabularies[field_index].get(value, unknown[field_index])
                    + offsets[field_index]
                )
            y[row_index] = row[6]
            users.append(row[1])
        encoded[name] = (X, y, users)
    return splits, encoded, int(sum(field_dims)), feature_names


def make_pair_groups(users, labels):
    groups = collections.defaultdict(lambda: [[], []])
    for index, (user, label) in enumerate(zip(users, labels)):
        groups[user][int(label)].append(index)
    usable = []
    for negatives, positives in groups.values():
        if negatives and positives:
            usable.append(
                (
                    np.asarray(positives, dtype=np.int64),
                    np.asarray(negatives, dtype=np.int64),
                )
            )
    return usable


def sample_pairs(
    pair_groups,
    X,
    model,
    rng,
    negative_per_positive=1,
    strategy="random",
    hard_candidates=5,
    hard_ratio=0.5,
    max_pairs=0,
):
    positive_parts = []
    negative_parts = []
    for positives, negatives in pair_groups:
        repeated = np.repeat(positives, negative_per_positive)
        if strategy == "random":
            selected = negatives[rng.integers(0, len(negatives), size=len(repeated))]
        elif strategy == "hard":
            random_selected = negatives[
                rng.integers(0, len(negatives), size=len(repeated))
            ]
            candidates = negatives[
                rng.integers(0, len(negatives), size=(len(repeated), hard_candidates))
            ]
            candidate_scores = model.predict(X[candidates.reshape(-1)]).reshape(
                len(repeated), hard_candidates
            )
            hard_selected = candidates[
                np.arange(len(repeated)), candidate_scores.argmax(1)
            ]
            use_hard = rng.random(len(repeated)) < hard_ratio
            selected = np.where(use_hard, hard_selected, random_selected)
        else:
            raise ValueError(f"Unknown negative strategy: {strategy}")
        positive_parts.append(repeated)
        negative_parts.append(selected)
    pos = np.concatenate(positive_parts)
    neg = np.concatenate(negative_parts)
    if max_pairs and len(pos) > max_pairs:
        keep = rng.choice(len(pos), size=max_pairs, replace=False)
        pos, neg = pos[keep], neg[keep]
    order = rng.permutation(len(pos))
    return pos[order], neg[order]


def run_experiment(args, training_mode, encoder_mode, negative_strategy="random"):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_log = output_dir / "epochs.jsonl"
    if epoch_log.exists():
        epoch_log.unlink()

    official_data, official_eval = load_official_modules(args.starter_dir)
    print(f"Loading data from {args.data_dir} ...")
    load_started = time.time()
    if encoder_mode == "official":
        splits, encoded, dim, feature_names = encode_official(
            args.data_dir, official_data
        )
    elif encoder_mode == "history":
        splits, encoded, dim, feature_names = encode_history(args.data_dir)
    else:
        raise ValueError(f"Unknown encoder mode: {encoder_mode}")
    print(
        f"Loaded in {time.time() - load_started:.1f}s | features={feature_names} | dim={dim}"
    )

    Xtr, ytr, utr = encoded["train"]
    Xva, yva, uva = encoded["valid"]
    Xte, yte, ute = encoded["test"]
    train_rows = splits["train"]
    valid_rows = splits["valid"]
    test_rows = splits["test"]
    smoke_test = bool(args.max_train_rows or args.max_eval_rows)
    if args.max_train_rows:
        limit = min(args.max_train_rows, len(Xtr))
        Xtr, ytr, utr = Xtr[:limit], ytr[:limit], utr[:limit]
        train_rows = train_rows[:limit]
    if args.max_eval_rows:
        limit = min(args.max_eval_rows, len(Xva))
        Xva, yva, uva = Xva[:limit], yva[:limit], uva[:limit]
        valid_rows = valid_rows[:limit]

    config = {
        "training_mode": training_mode,
        "encoder_mode": encoder_mode,
        "negative_strategy": negative_strategy,
        "features": feature_names,
        "k": args.k,
        "lr": args.lr,
        "l2": args.l2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
        "negative_per_positive": args.negative_per_positive,
        "hard_candidates": args.hard_candidates,
        "hard_negative_warmup": args.hard_negative_warmup,
        "hard_negative_ratio": args.hard_negative_ratio,
        "max_pairs_per_epoch": args.max_pairs_per_epoch,
        "train_rows": len(Xtr),
        "valid_rows": len(Xva),
        "smoke_test": smoke_test,
    }
    write_json(output_dir / "config.json", config)

    model = FM(dim, k=args.k, lr=args.lr, l2=args.l2, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    pair_groups = None
    if training_mode == "pairwise":
        pair_groups = make_pair_groups(utr, ytr)
        if not pair_groups:
            raise RuntimeError("No users with both positive and negative rows")
        print(f"Pair-eligible users: {len(pair_groups):,}")

    best_score = -1.0
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    total_started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        losses = []
        examples = 0
        if training_mode == "pointwise":
            order = rng.permutation(len(ytr))
            for start in range(0, len(order), args.batch_size):
                index = order[start : start + args.batch_size]
                losses.append(model.step_pointwise(Xtr[index], ytr[index]))
                examples += len(index)
        elif training_mode == "pairwise":
            epoch_negative_strategy = (
                "random"
                if negative_strategy == "hard"
                and epoch <= args.hard_negative_warmup
                else negative_strategy
            )
            pos, neg = sample_pairs(
                pair_groups,
                Xtr,
                model,
                rng,
                negative_per_positive=args.negative_per_positive,
                strategy=epoch_negative_strategy,
                hard_candidates=args.hard_candidates,
                hard_ratio=args.hard_negative_ratio,
                max_pairs=args.max_pairs_per_epoch,
            )
            for start in range(0, len(pos), args.batch_size):
                p = pos[start : start + args.batch_size]
                n = neg[start : start + args.batch_size]
                losses.append(model.step_pairwise(Xtr[p], Xtr[n]))
                examples += len(p)
        else:
            raise ValueError(f"Unknown training mode: {training_mode}")

        valid_scores = model.predict(Xva)
        valid_metrics = official_eval.evaluate(uva, yva, valid_scores)
        elapsed = time.time() - epoch_started
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "examples_or_pairs": examples,
            "negative_strategy": (
                "none"
                if training_mode == "pointwise"
                else epoch_negative_strategy
            ),
            "valid": {
                "GAUC": float(valid_metrics["GAUC"]),
                "nDCG@5": float(valid_metrics["nDCG@5"]),
                "primary": float(valid_metrics["primary"]),
            },
            "elapsed_seconds": round(elapsed, 3),
        }
        append_jsonl(epoch_log, record)
        print(
            f"epoch {epoch:02d} | loss {record['loss']:.4f} | "
            f"valid GAUC {record['valid']['GAUC']:.4f} | "
            f"nDCG@5 {record['valid']['nDCG@5']:.4f} | "
            f"primary {record['valid']['primary']:.4f} | {elapsed:.1f}s"
        )
        if valid_metrics["primary"] > best_score + 1e-5:
            best_score = float(valid_metrics["primary"])
            best_epoch = epoch
            best_state = model.state()
            model.save(output_dir / "best_model.npz")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch}; best epoch was {best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.restore(best_state)
    valid_scores = model.predict(Xva)
    valid_metrics = official_eval.evaluate(uva, yva, valid_scores)
    validation_predictions = output_dir / "validation_predictions.csv"
    write_predictions(validation_predictions, valid_rows, valid_scores)
    validate_predictions(validation_predictions, valid_rows)

    test_metrics = None
    if args.score_test:
        test_scores = model.predict(Xte)
        test_metrics = official_eval.evaluate(ute, yte, test_scores)
        write_predictions(output_dir / "test_predictions.csv", test_rows, test_scores)

    summary = {
        "status": "smoke_test" if smoke_test else "complete",
        "best_epoch": best_epoch,
        "valid": {
            "GAUC": float(valid_metrics["GAUC"]),
            "nDCG@5": float(valid_metrics["nDCG@5"]),
            "primary": float(valid_metrics["primary"]),
            "delta_vs_official_valid_fm": (
                None
                if smoke_test
                else float(valid_metrics["primary"] - OFFICIAL_VALID_PRIMARY)
            ),
        },
        "test": (
            None
            if test_metrics is None
            else {
                "GAUC": float(test_metrics["GAUC"]),
                "nDCG@5": float(test_metrics["nDCG@5"]),
                "primary": float(test_metrics["primary"]),
            }
        ),
        "runtime_seconds": round(time.time() - total_started, 3),
        "config": config,
        "artifacts": {
            "checkpoint": "best_model.npz",
            "epoch_log": "epochs.jsonl",
            "validation_predictions": "validation_predictions.csv",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print("\nFinal validation:")
    print(json.dumps(summary["valid"], ensure_ascii=False, indent=2))
    print(f"Artifacts: {output_dir}")
    return summary
