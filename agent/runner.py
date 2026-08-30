from __future__ import annotations

from typing import Any, Dict


class DryRunRunner:
    def run(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Dry-run runner requires a plan dictionary.")

        metrics = plan.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Plan {plan.get('experiment_id', 'unknown')} has malformed metrics.")

        ga = metrics.get("GAUC")
        ndcg = metrics.get("nDCG@5")
        if ga is None or ndcg is None:
            raise ValueError(f"Plan {plan.get('experiment_id', 'unknown')} is missing GAUC or nDCG@5.")

        result = {
            "GAUC": float(ga),
            "nDCG@5": float(ndcg),
            "primary": float(metrics.get("primary", 0.5 * (float(ga) + float(ndcg)))),
            "dry_run": True,
            "validation": {
                "GAUC": float(ga),
                "nDCG@5": float(ndcg),
                "primary": float(metrics.get("primary", 0.5 * (float(ga) + float(ndcg)))),
                "dry_run": True,
            },
        }
        return result


def dry_run_run(plan: Dict[str, Any]) -> Dict[str, Any]:
    return DryRunRunner().run(plan)
