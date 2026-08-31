import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FIELDS = (
    "experiment_id",
    "parent_id",
    "hypothesis",
    "configuration",
    "seed",
    "command",
    "git_revision",
    "metrics",
    "status",
    "decision",
    "error",
    "recovery",
    "wall_clock_sec",
    "llm_tokens",
    "manual_interventions",
    "timestamp",
)


class ExperimentLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(record, dict):
            raise ValueError("Experiment log record must be a dictionary.")

        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

        # Keep the record shape strict but allow the values to be empty or null where appropriate.
        for field in REQUIRED_FIELDS:
            if record.get(field) is None and field not in {"parent_id", "error", "recovery"}:
                raise ValueError(f"Field '{field}' cannot be null.")

        if not isinstance(record["configuration"], dict):
            raise ValueError("Field 'configuration' must be a dictionary.")
        if not isinstance(record["metrics"], dict):
            raise ValueError("Field 'metrics' must be a dictionary.")
        if not isinstance(record["llm_tokens"], dict):
            raise ValueError("Field 'llm_tokens' must be a dictionary.")
        if not isinstance(record["manual_interventions"], list):
            raise ValueError("Field 'manual_interventions' must be a list.")

        return record

    def append(self, record: Dict[str, Any]) -> Path:
        validated = self._validate_record(record)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(validated, sort_keys=True) + "\n")
        return self.log_path

    def load(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records
