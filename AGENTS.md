# Repository Rules

## Project Goal

- Build a reproducible Autonomous ML Research Agent for the KuaiRand recommender-system task.
- Codex is the development assistant. The final project agent must run independently from one command.
- The agent must propose, run, evaluate, accept or reject, recover from, and log experiments.

## Protected Competition Contracts

- Never modify `evaluate.py`.
- Never change `LABEL = 'long_view'`.
- Never change the official train, validation, or test date splits.
- Never change the order in which official CSV rows are read or evaluated.
- Never change the official submission header, `row_id` alignment, row count, or numeric-score requirements.
- Preserve `baseline.py` as the reproducible official baseline.
- Preserve `baseline_scores.json` as benchmark metadata.
- Implement research models in new modules rather than overwriting official baseline behavior.
- `README.md` may be updated for project documentation when explicitly allowed.

## Evaluation Policy

- Use validation metrics only for routine experiment selection.
- Do not repeatedly use test metrics for model selection.
- Test evaluation and final submission require an explicit final-stage action.
- Primary score is the arithmetic mean of GAUC and nDCG@5.
- Do not implement replacement metrics or unofficial evaluation logic.

## Data-Leakage Rules

- Fit vocabularies, statistics, quantiles, encoders, sampling distributions, and preprocessing on training data only.
- Historical features must use only events before the target event.
- Same-row outcome fields must never be inference features.
- Preserve a unique row identity for repeated user-video impressions.
- Pairwise BPR positive and negative samples must come from the same user.
- Do not use validation or test outcomes when constructing BPR training pairs.

## Repository Structure

Prefer this structure:

```text
agent/
experiments/
configs/
tests/
logs/
artifacts/
run_agent.py
```

- `agent/` contains orchestration components.
- `experiments/` contains isolated model or feature experiments.
- `configs/` contains reproducible experiment configurations.
- `tests/` contains contract, leakage, sampling, and regression tests.
- `logs/` contains structured experiment records.
- `artifacts/` contains generated checkpoints and reports and must remain ignored by Git.

## Implementation Order

- E001: Preserve and reproduce the official FM baseline.
- E002: Add repository guardrails, contract tests, experiment configuration, and structured logging.
- E003: Build a minimal rule-based autonomous loop: `planner -> runner -> evaluator -> decision -> logger`.
- E004: Implement pairwise BPR-FM as the first research experiment plugin.
- E005: Add timeout, failure recovery, rollback, best-checkpoint tracking, and convergence.
- E006+: Add listwise loss, causal history features, auxiliary objectives, and other research directions.
- Final: Upgrade the planner to use a callable LLM interface with structured JSON output.

## Experiment Discipline

- Every experiment must have an `experiment_id` and one primary hypothesis.
- Every run must record `parent_id`, hypothesis, configuration, seed, command, Git revision, validation metrics, decision, error, recovery, wall-clock time, token usage, and manual interventions.
- Change one main research variable per experiment.
- A score regression is a normal `REJECT` decision, not necessarily a software error.
- Verify promising improvements using multiple fixed seeds.
- Apply the documented convergence rule: stop after three consecutive iterations without a validation-primary improvement greater than `0.002`.

## Codex Working Rules

- Inspect existing code before editing.
- For multi-file or model changes, provide a plan before editing.
- Make the smallest change required for the current task.
- Modify only explicitly allowed files.
- Do not install new dependencies without approval.
- Do not run network commands without approval.
- Never use destructive commands such as `git reset --hard` or recursive deletion.
- Never expose, print, or commit API keys, `.env` files, raw datasets, virtual environments, checkpoints, or secret-like files.
- Do not push, merge, or open a pull request unless explicitly requested.

## Verification

After relevant changes:

- Run `python -m compileall .`.
- Run `python -m unittest discover -s tests` when tests exist.
- Run only the relevant baseline or validation command.
- Show `git status`.
- Show `git diff --stat`.
- Summarize every changed file and the verification results.
- Confirm that protected competition contracts remain unchanged.
