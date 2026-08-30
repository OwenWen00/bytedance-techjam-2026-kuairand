import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.fm_validation import main


class FMValidationWrapperTests(unittest.TestCase):
    def test_wrapper_uses_train_rows_only_and_valid_scores_only(self):
        rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u1", "v2", "a2", "tab1", 200.0, 0),
            (20220422, "u1", "v3", "a3", "tab2", 150.0, 1),
            (20220422, "u1", "v4", "a4", "tab2", 180.0, 0),
            (20220422, "u2", "v5", "a5", "tab1", 120.0, 0),
            (20220422, "u2", "v6", "a6", "tab1", 170.0, 1),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            train_path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
            valid_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
            train_path.write_text(
                "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
                "20220408,u1,v1,a1,tab1,100,1\n"
                "20220408,u1,v2,a2,tab1,200,0\n",
                encoding="utf-8",
            )
            valid_path.write_text(
                "date,user_id,video_id,author_id,tab,duration_ms,long_view\n"
                "20220422,u1,v3,a3,tab2,150,1\n"
                "20220422,u1,v4,a4,tab2,180,0\n"
                "20220422,u2,v5,a5,tab1,120,0\n"
                "20220422,u2,v6,a6,tab1,170,1\n",
                encoding="utf-8",
            )
            video_features = data_dir / "video_features_basic_pure.csv"
            video_features.write_text("video_id,author_id\nv1,a1\nv2,a2\nv3,a3\nv4,a4\nv5,a5\nv6,a6\n", encoding="utf-8")

            with patch("experiments.fm_validation.load", return_value={"train": rows[:2], "valid": rows[2:], "test": rows}), patch(
                "experiments.fm_validation.encode",
                return_value=({"train": (Mock(), Mock(), ["u1", "u1"]), "valid": (Mock(), Mock(), ["u1", "u1", "u2", "u2"])}, 16),
            ), patch("experiments.fm_validation.evaluate", return_value={"GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65}) as evaluate_mock:
                result_file = Path(tmpdir) / "result.json"
                with patch("sys.argv", ["fm_validation.py", "--experiment-id", "E003_fm_repro", "--seed", "0", "--k", "16", "--lr", "0.001", "--batch", "8192", "--max-epochs", "40", "--patience", "4", "--data-dir", str(data_dir), "--result-path", str(result_file)]):
                    main()

                payload = json.loads(result_file.read_text(encoding="utf-8"))
                self.assertEqual(payload["experiment_id"], "E003_fm_repro")
                self.assertEqual(payload["split"], "validation")
                self.assertNotIn("primary", payload)
                self.assertNotIn("test", json.dumps(payload).lower())
                evaluate_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
