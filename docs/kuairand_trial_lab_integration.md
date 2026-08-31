# KuaiRand trial-lab integration

## Scope

The local `kuairand_trial_lab` model implementation is now part of this repository under `models/`. It remains NumPy-only and imports the repository's frozen `data.py` and `evaluate.py` read-only. No personal path is encoded in the model or Agent.

The production adapter is `agent.kuairand`. It registers four explicit tools:

- `run_pointwise_fm`
- `run_pairwise_bpr`
- `run_hard_negative_bpr`
- `run_history_pairwise`

Each tool accepts only its declared numeric parameters. The subprocess command is built as an argv list, the data directory is injected at runtime and redacted from the ledger, and outputs are isolated by run/iteration/attempt. `score-test`, partial-row smoke parameters, arbitrary script paths, and free-form commands are not registered Agent parameters.

Every non-baseline deterministic plan also creates one run-scoped `E###-active-variant.json` through the controlled patcher. The tool refuses to run unless that applied diff matches its registered variant. A rejected experiment removes the marker through exact rollback; an accepted experiment keeps it as the selected config evidence. This makes the demo exercise a real planner-triggered config diff without mutating model source or any frozen official file.

## Agent decision path

The deterministic no-key planner starts with the comparable pointwise baseline. It then prefers the lower-cost isolated BPR loss change before the history feature bundle. This supports the intended three-step demo:

1. Establish pointwise FM baseline.
2. Evaluate pairwise BPR; keep it as the validation best if its primary is strictly higher.
3. Evaluate leakage-safe history/time Pairwise; likewise keep any strict new best while measuring cumulative progress against the convergence reference.

Hard-negative BPR remains registered for an LLM-authored plan, but it is intentionally absent from the deterministic queue because the existing evidence rejected the tested configuration. This prevents the fallback planner from spending budget repeating a known non-improvement.

The three canonical comparisons are only the initialization stage. If the orchestration budget and convergence rule still allow work, the planner selects the highest accepted validation result and generates one-parameter neighbors of its exact executed configuration. Learning rate, regularization, pair count/sampling, batch size, patience, and epoch-budget values come from a declared per-tool search space. The planner interleaves dimensions rather than exhausting one dimension first, and configuration fingerprints prevent repeats even when a plan has a different natural-language hypothesis. Any strict new validation best becomes the next search anchor; epsilon is reserved for convergence significance rather than checkpoint selection.

Every result records `params`, `seed`, `feature_flags`, and `plan_fingerprint`. This is especially important after deterministic recovery because the ledger records the configuration that actually executed, not just the original requested plan. It also records `planner_source`, `planner_error`, and a bounded planner response excerpt, making LLM-to-deterministic fallback directly auditable. The default convergence definition is three consecutive iterations without a cumulative gain greater than `0.002` from the last convergence reference; longer exploratory runs must raise `--convergence-patience` explicitly while remaining bounded by 50 iterations and six hours.

## Existing validation evidence

The pre-integration trial lab reported official full-validation primary scores over three paired seeds:

| Variant | seed 0 | seed 1 | seed 2 | mean | mean paired delta vs pointwise |
|---|---:|---:|---:|---:|---:|
| Pointwise FM | 0.601470 | 0.601761 | 0.601090 | 0.601440 | 0 |
| Pairwise BPR | 0.603396 | 0.602221 | 0.603226 | 0.602948 | +0.001507 |
| History/time Pairwise | 0.603638 | 0.603143 | 0.604199 | 0.603660 | +0.002220 |

These are validation-only model-selection results. They do not claim a hidden-test or online result. Integrated runs independently write their config, epoch log, best checkpoint, aligned validation predictions, summary, stdout and stderr into the Agent experiment record.

## Integrated acceptance run

Run `integrated-model-evidence` executed all three full-data iterations from implementation commit `93871d148fc73d0ad8bcbe2b26c041f6bb7ce415`:

| Iteration | Decision | GAUC | nDCG@5 | Primary | delta vs accepted best |
|---|---|---:|---:|---:|---:|
| Pointwise FM | accepted baseline | 0.667133 | 0.535806 | 0.601470 | — |
| Pairwise BPR | rejected | 0.669711 | 0.537082 | 0.603396 | +0.001927 |
| History/time Pairwise | accepted | 0.669704 | 0.537571 | 0.603638 | +0.002168 |

The run used `token_usage=0` and `gpu_hours=0.0`, stopped at the configured three-iteration bound, and never scored test. The rejected BPR marker was removed; the accepted history marker was kept. The accepted validation prediction file passed the frozen `submit.py --check --split valid` check for all 124,909 rows.

## Direct connectivity check

The direct model CLI has a smoke mode for code connectivity only. It cannot be selected through the Agent tool:

```bash
python3 -m models.run_trial \
  --variant history_pairwise \
  --data-dir /path/to/KuaiRand-Pure/data \
  --output-dir /tmp/kuairand-history-smoke \
  --smoke
```

For a production research run, omit `--smoke` and launch through `python3 -m agent` so plan validation, timeout, recovery, selection, rollback and the append-only ledger remain active.
