from typing import Optional


def choose_decision(candidate_primary: float, current_best_primary: Optional[float]) -> str:
    if current_best_primary is None:
        return "ACCEPT"
    return "ACCEPT" if float(candidate_primary) > float(current_best_primary) else "REJECT"
