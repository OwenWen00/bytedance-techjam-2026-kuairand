import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.fm_history_validation import main


class HistoryFMTests(unittest.TestCase):
    def test_history_fm_emits_validation_metrics_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            train_path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
            valid_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
            train_path.write_text(
                "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
                "20220408,u1,v1,a1,tab1,100,1\n"
                "20220408,u1,v2,a1,tab1,200,0\n"
                "20220408,u2,v3,a2,tab1,150,1\n",
                encoding="utf-8",
            )
            valid_path.write_text(
                "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
                "20220422,u1,v4,a1,tab2,120,1\n"
                "20220422,u2,v5,a2,tab2,140,0\n",
                encoding="utf-8",
            )
            video_features = data_dir / "video_features_basic_pure.csv"
            video_features.write_text(
                "video_id,author_id\n"
                "v1,a1\n"
                "v2,a1\n"
                "v3,a2\n"
                "v4,a1\n"
                "v5,a2\n",
                encoding="utf-8",
            )
            result_path = Path(tmpdir) / "result.json"
            with patch("sys.argv", [
                "fm_history_validation.py",
                "--experiment-id",
                "history_fm",
                "--seed",
                "0",
                "--k",
                "16",
                "--lr",
                "0.001",
                "--batch",
                "8192",
                "--max-epochs",
                "2",
                "--patience",
                "1",
                "--data-dir",
                str(data_dir),
                "--result-path",
                str(result_path),
            ]):
                main()

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], "history_fm")
            self.assertEqual(payload["split"], "validation")
            self.assertNotIn("primary", payload)
            self.assertNotIn("primary", payload["metrics"])
            self.assertIn("GAUC", payload["metrics"])
            self.assertIn("nDCG@5", payload["metrics"])
            self.assertNotIn("test", json.dumps(payload).lower())


if __name__ == "__main__":
    unittest.main()
