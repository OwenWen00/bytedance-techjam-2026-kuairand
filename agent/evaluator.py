import math
from typing import Any, Dict


def evaluate_agent_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("Metrics payload must be a dictionary.")

    if "GAUC" in metrics:
        gauc = metrics["GAUC"]
        ndcg = metrics["nDCG@5"]
    elif "validation" in metrics and isinstance(metrics["validation"], dict):
        validation = metrics["validation"]
        gauc = validation.get("GAUC")
        ndcg = validation.get("nDCG@5")
    else:
        raise ValueError("Malformed metric dictionary: missing GAUC and nDCG@5.")

    try:
        gauc_value = float(gauc)
        ndcg_value = float(ndcg)
    except (TypeError, ValueError) as exc:
        raise ValueError("Malformed metric dictionary: GAUC and nDCG@5 must be numeric.") from exc

    if not math.isfinite(gauc_value) or not math.isfinite(ndcg_value):
        raise ValueError("Malformed metric dictionary: GAUC and nDCG@5 must be finite numbers.")

    primary = 0.5 * (gauc_value + ndcg_value)
    return {
        "GAUC": gauc_value,
        "nDCG@5": ndcg_value,
        "primary": primary,
        "dry_run": bool(metrics.get("dry_run", False) or (isinstance(metrics.get("validation"), dict) and bool(metrics["validation"].get("dry_run", False)))),
    }
