import os
import sys
import tempfile
import time
import unittest

from agent.runner import CommandTimeout, RunnerError, SafeRunner


class RunnerTests(unittest.TestCase):
    def test_timeout_is_bounded_and_logs_are_redacted(self):
        with tempfile.TemporaryDirectory() as root:
            runner = SafeRunner(root, redactions={"top-secret": "<SECRET>"})
            started = time.monotonic()
            with self.assertRaises(CommandTimeout) as caught:
                runner.run(
                    [sys.executable, "-c", "import time; print('token=top-secret', flush=True); time.sleep(5)"],
                    0.15, os.path.join(root, "logs"),
                )
            self.assertLess(time.monotonic() - started, 3.0)
            self.assertNotIn("top-secret", caught.exception.result.stdout_summary)

    def test_nonzero_exit_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RunnerError) as caught:
                SafeRunner(root).run([sys.executable, "-c", "raise SystemExit(7)"], 2, root)
            self.assertEqual(7, caught.exception.result.returncode)

    def test_full_log_is_preserved_while_summary_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            result = SafeRunner(root, max_summary_chars=10).run(
                [sys.executable, "-c", "print('x' * 100)"], 2, root,
            )
            self.assertIn("TRUNCATED", result.stdout_summary)
            with open(result.stdout_path, encoding="utf-8") as handle:
                self.assertGreater(len(handle.read()), 90)


if __name__ == "__main__":
    unittest.main()
