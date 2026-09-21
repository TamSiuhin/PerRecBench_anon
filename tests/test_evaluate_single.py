"""Regression checks for Kendall tau on ranked lists of user IDs.

Run from the repository root with:
    python3 -B -m unittest discover -s tests -p 'test_*.py' -v
"""

import importlib.util
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "evaluate_single.py"
SPEC = importlib.util.spec_from_file_location("evaluate_single", SCRIPT)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def pairwise_tau(predicted_order, true_order):
    """Independent oracle: count concordant and discordant user pairs."""
    predicted_positions = {user: i for i, user in enumerate(predicted_order)}
    true_positions = {user: i for i, user in enumerate(true_order)}
    concordant = 0
    discordant = 0
    for left, right in itertools.combinations(true_order, 2):
        same_order = (
            predicted_positions[left] < predicted_positions[right]
        ) == (true_positions[left] < true_positions[right])
        if same_order:
            concordant += 1
        else:
            discordant += 1
    return (concordant - discordant) / (concordant + discordant)


class KendallTauRegressionTests(unittest.TestCase):
    def setUp(self):
        evaluate.error2 = 0

    def test_ranks_are_aligned_by_user_identity(self):
        np.testing.assert_array_equal(
            evaluate.ranking_to_ranks([3, 1, 2]), [1, 2, 0]
        )

    def test_nontrivial_order_regression(self):
        # Directly correlating these ID lists gave -1 rather than +1/3.
        actual = evaluate.compute_tau({"item": [1, 3, 2]}, {"item": [3, 1, 2]})
        self.assertAlmostEqual(actual, 1 / 3)

    def test_identical_and_reversed_rankings(self):
        for order in ([2, 1], [3, 1, 2], [2, 4, 1, 3]):
            with self.subTest(order=order):
                self.assertAlmostEqual(
                    evaluate.compute_tau({"item": order}, {"item": order}), 1.0
                )
                self.assertAlmostEqual(
                    evaluate.compute_tau(
                        {"item": list(reversed(order))}, {"item": order}
                    ),
                    -1.0,
                )

    def test_all_small_permutations_match_pair_counting(self):
        # 4 + 36 + 576 = 616 combinations; also covers every two-user case.
        for size in range(2, 5):
            orders = list(itertools.permutations(range(1, size + 1)))
            for predicted, truth in itertools.product(orders, repeat=2):
                with self.subTest(size=size, predicted=predicted, truth=truth):
                    self.assertAlmostEqual(
                        evaluate.compute_tau({"item": predicted}, {"item": truth}),
                        pairwise_tau(predicted, truth),
                    )

    def test_tau_is_invariant_to_user_id_relabeling(self):
        predicted = [1, 3, 2, 4]
        truth = [3, 1, 4, 2]
        expected = pairwise_tau(predicted, truth)
        for new_ids in itertools.permutations(range(1, 5)):
            mapping = dict(zip(range(1, 5), new_ids))
            with self.subTest(new_ids=new_ids):
                self.assertAlmostEqual(
                    evaluate.compute_tau(
                        {"item": [mapping[user] for user in predicted]},
                        {"item": [mapping[user] for user in truth]},
                    ),
                    expected,
                )

    def test_invalid_rankings_are_rejected(self):
        invalid_orders = (
            [],
            [1],
            [1, 1, 3],
            [0, 1, 2],
            [1, 2, 4],
            [1, 2, 2.5],
            [1, 2, float("nan")],
            [[1, 2, 3]],
            "format_not_correct",
        )
        for order in invalid_orders:
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    evaluate.ranking_to_ranks(order)

    def test_invalid_prediction_is_skipped_and_counted(self):
        actual = evaluate.compute_tau(
            {"valid": [3, 1, 2], "invalid": [1, 1, 3]},
            {"valid": [3, 1, 2], "invalid": [3, 1, 2]},
        )
        self.assertAlmostEqual(actual, 1.0)
        self.assertEqual(evaluate.error2, 1)


class EvaluatorCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.label_path = self.directory / "labels.json"
        self.label_path.write_text(
            json.dumps(
                {
                    "item": [
                        [
                            {"relative_rating": 1.0, "avg_rating": 3.0, "rating": 4},
                            {"relative_rating": -1.0, "avg_rating": 2.0, "rating": 1},
                            {"relative_rating": 2.0, "avg_rating": 1.0, "rating": 3},
                        ]
                    ]
                }
            ),
            encoding="utf-8",
        )

    def run_evaluator(self, mode, prediction, relative=False):
        prediction_path = self.directory / "predictions.json"
        prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
        command = [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--pred_path",
            str(prediction_path),
            "--label_path",
            str(self.label_path),
            "--mode",
            mode,
        ]
        if relative:
            command.append("--rel_rating")
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        match = re.search(r"^Kendall-Tau:\s*(\S+)", process.stdout, re.MULTILINE)
        self.assertIsNotNone(match, process.stdout)
        return float(match.group(1))

    def test_point_absolute_ratings_are_centered_before_ranking(self):
        # Centered scores [0.8, -0.5, 0.4] rank users [1, 3, 2].
        actual = self.run_evaluator("point", {"item": [[3.8, 1.5, 1.4]]})
        self.assertAlmostEqual(actual, 1 / 3)

    def test_point_relative_ratings_are_not_centered_again(self):
        actual = self.run_evaluator(
            "point", {"item": [[0.8, -0.5, 0.4]]}, relative=True
        )
        self.assertAlmostEqual(actual, 1 / 3)

    def test_pair_ranked_user_ids(self):
        actual = self.run_evaluator("pair", {"item": [[[1, 3, 2]]]})
        self.assertAlmostEqual(actual, 1 / 3)

    def test_group_ranked_user_ids(self):
        actual = self.run_evaluator("group", {"item": [[[1, 3, 2]]]})
        self.assertAlmostEqual(actual, 1 / 3)

    def test_point_tie_breaking_preserves_existing_argsort_protocol(self):
        # This patch fixes the tau representation, not the pointwise tie policy.
        predicted_order = (np.argsort([0.0, 0.0, 0.0]) + 1)[::-1].tolist()
        actual = self.run_evaluator(
            "point", {"item": [[0.0, 0.0, 0.0]]}, relative=True
        )
        self.assertAlmostEqual(actual, pairwise_tau(predicted_order, [3, 1, 2]))


if __name__ == "__main__":
    unittest.main()
