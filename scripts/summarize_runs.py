"""Print a compact table from one or more append-only experiment ledgers."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="experiments.jsonl files")
    args = parser.parse_args()
    paths = [Path(value) for value in args.paths]
    if not paths:
        paths = sorted(Path("experiments/logs").glob("*/experiments.jsonl"))
    print("run\titeration\tstatus\tprimary\tdelta\tattempt\tseconds\ttokens\tgpu_hours\tchange")
    for path in paths:
        best = None
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            primary = record.get("primary")
            delta = None if primary is None or best is None else primary - best
            if record["status"] == "accepted" and primary is not None:
                best = primary
            print("%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%s\t%s\t%s" % (
                record["run_id"], record["iteration"], record["status"],
                "null" if primary is None else "%.6f" % primary,
                "null" if delta is None else "%.6f" % delta,
                record.get("attempt", 1), record.get("elapsed_seconds", 0.0),
                record.get("token_usage", "null"), record.get("gpu_hours", "null"),
                record.get("single_primary_change", ""),
            ))


if __name__ == "__main__":
    main()
