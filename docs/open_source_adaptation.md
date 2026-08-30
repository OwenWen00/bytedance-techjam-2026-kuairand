# Open-source adaptation boundaries

The agent layer is an original, standard-library implementation. It does not copy source code from the projects below.

## karpathy/autoresearch

Adapted concepts: keep the evaluator immutable, restrict the editable surface, impose a fixed experiment budget, and use an explicit change -> evaluate -> keep/revert loop. This project generalizes those ideas through `AgentPolicy`, `ControlledPatcher`, `ValidationSelector`, and `Orchestrator`; it does not inherit autoresearch's GPU, model, or single-file assumptions. Repository: https://github.com/karpathy/autoresearch (MIT, checked 2026-08-30).

## VectorInstitute/helix

Adapted concepts: machine-readable experiment contracts, explicit editable/read-only scope, a registered evaluation command, Git/file provenance, and an append-only experiment ledger. This project uses validated Python dataclasses and JSON/JSONL because Python 3.9 and the standard library are the local compatibility baseline. Repository: https://github.com/VectorInstitute/helix (Apache-2.0, checked 2026-08-30).

## WecoAI/weco-cli

Adapted only at the strategy level: experiment selection should use evaluation evidence and preserve failed branches as useful research evidence. No external service, account, or cloud runtime is required by this implementation. Repository: https://github.com/WecoAI/weco-cli (checked 2026-08-30).

## Deliberate exclusions

This project does not implement a web UI, container platform, or tree-search infrastructure. Its integrated NumPy model is isolated behind the same trusted driver contract used by future model teams; OpenAI and DeepSeek are optional planner providers and never receive command-execution authority.
