"""Unit tests for the paired statistics and answer-quality metrics.

Run: python -m unittest tests.test_evaluation_metrics
"""

import unittest

from src.evaluation.stats import (
    paired_randomization_test, holm, wilson_ci, roc_auc, bootstrap_ci, compare_configs,
)
from src.evaluation.answer_quality_eval import tokens, token_overlap, rouge_l, lcs_length


class TestStats(unittest.TestCase):

    def test_identical_systems_p_is_one(self):
        self.assertEqual(paired_randomization_test([0.5, 0.2, 0.9], [0.5, 0.2, 0.9]), 1.0)

    def test_consistent_small_gain_is_significant_when_paired(self):
        # Large between-question spread, constant +0.05 gain: only a paired
        # test can see it.
        base = [0.1, 0.9, 0.3, 0.7, 0.5] * 6
        better = [b + 0.05 for b in base]
        self.assertLess(paired_randomization_test(better, base), 0.01)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.9]), [0.9])

    def test_wilson_ci_contains_point_estimate(self):
        lo, hi = wilson_ci(7, 13)
        self.assertLess(lo, 7 / 13)
        self.assertGreater(hi, 7 / 13)
        self.assertAlmostEqual(lo, 0.291, places=3)
        self.assertAlmostEqual(hi, 0.768, places=3)

    def test_wilson_ci_zero_successes(self):
        lo, hi = wilson_ci(0, 35)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.099, places=3)

    def test_roc_auc(self):
        self.assertEqual(roc_auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(roc_auc([0.1], [0.9]), 0.0)
        self.assertEqual(roc_auc([0.5], [0.5]), 0.5)

    def test_bootstrap_ci_brackets_mean(self):
        lo, hi = bootstrap_ci([0.0, 1.0] * 20)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_compare_configs_diff_sign(self):
        rows = compare_configs({"a": {"q1": 1.0, "q2": 1.0}, "b": {"q1": 0.0, "q2": 0.5}}, "m")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["diff"], 0.75)  # group1 - group2
        self.assertEqual(rows[0]["wins"], 2)


class TestAnswerQuality(unittest.TestCase):

    def test_tokens_strip_citations_and_stop_words(self):
        self.assertEqual(tokens("You can work 48 hours [S03_002]."), ["work", "48", "hours"])

    def test_token_overlap(self):
        p, r, f = token_overlap(["work", "48", "hours", "fortnight"], ["48", "hours"])
        self.assertEqual((p, r), (0.5, 1.0))
        self.assertAlmostEqual(f, 2 / 3)

    def test_no_overlap(self):
        self.assertEqual(token_overlap(["a"], ["b"]), (0.0, 0.0, 0.0))

    def test_lcs(self):
        self.assertEqual(lcs_length(list("abcde"), list("ace")), 3)
        p, r, _ = rouge_l(list("abcde"), list("ace"))
        self.assertEqual((p, r), (0.6, 1.0))


if __name__ == "__main__":
    unittest.main()
