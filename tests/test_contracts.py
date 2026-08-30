import unittest

import data
from evaluate import evaluate


class CompetitionContractTests(unittest.TestCase):
    def test_label_and_official_date_splits_remain_fixed(self):
        self.assertEqual(data.LABEL, "long_view")
        self.assertEqual(
            data.SPLITS,
            {
                "train": (20220408, 20220421),
                "valid": (20220422, 20220428),
                "test": (20220429, 20220508),
            },
        )

    def test_official_evaluator_handles_mixed_zero_positive_and_tied_scores(self):
        result = evaluate(
            user_ids=["mixed", "mixed", "zero", "zero"],
            labels=[1, 0, 0, 0],
            scores=[0.5, 0.5, 0.9, 0.1],
        )

        self.assertAlmostEqual(result["GAUC"], 0.5)
        self.assertAlmostEqual(result["nDCG@5"], 0.5)
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["rows"], 4)

    def test_primary_is_arithmetic_mean_of_official_metrics(self):
        result = evaluate(
            user_ids=["mixed", "mixed", "zero"],
            labels=[0, 1, 0],
            scores=[0.8, 0.2, 0.4],
        )

        expected_primary = 0.5 * (result["GAUC"] + result["nDCG@5"])
        self.assertAlmostEqual(result["primary"], expected_primary)


if __name__ == "__main__":
    unittest.main()
