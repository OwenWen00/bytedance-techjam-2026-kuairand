import tempfile
import unittest
from pathlib import Path

from agent.logger import ExperimentLogger


class ExperimentLoggerTests(unittest.TestCase):
    def _make_record(self, **overrides):
        record = {
            "experiment_id": "exp_001",
            "parent_id": None,
            "hypothesis": "The baseline should remain stable.",
            "configuration": {
                "name": "E001_fm",
                "split": "validation",
                "model": "fm_baseline",
            },
            "seed": 7,
            "command": "python baseline.py",
            "git_revision": "abc123",
            "metrics": {"validation": {"gauc": 0.5, "ndcg@5": 0.4}},
            "status": "RUNNING",
            "decision": "PENDING",
            "error": None,
            "recovery": None,
            "wall_clock_sec": 12.5,
            "llm_tokens": {"prompt": 10, "completion": 5, "total": 15},
            "manual_interventions": [],
            "timestamp": "2026-08-30T12:00:00Z",
        }
        record.update(overrides)
        return record

    def test_append_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment.jsonl"
            logger = ExperimentLogger(path)

            rec1 = self._make_record(experiment_id="exp_001")
            rec2 = self._make_record(experiment_id="exp_002", parent_id="exp_001")

            logger.append(rec1)
            logger.append(rec2)
            loaded = logger.load()

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["experiment_id"], "exp_001")
            self.assertEqual(loaded[1]["experiment_id"], "exp_002")
            self.assertEqual(loaded[1]["parent_id"], "exp_001")

    def test_missing_required_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiment.jsonl"
            logger = ExperimentLogger(path)
            bad_record = self._make_record()
            bad_record.pop("decision")

            with self.assertRaisesRegex(ValueError, "Missing required field"):
                logger.append(bad_record)

            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
