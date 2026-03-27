from __future__ import annotations

import csv
import unittest
from pathlib import Path

from tennis_betting.backtest import no_vig_probabilities, run_backtest
from tennis_betting.config import BacktestConfig
from tennis_betting.data import load_matches


CSV_HEADER = ["Date", "Surface", "Winner", "Loser", "PSW", "PSL", "Comment"]
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = WORKSPACE_ROOT / "data" / "processed" / "test_matches.csv"


class BacktestTests(unittest.TestCase):
    def test_no_vig_probabilities_sum_to_one(self) -> None:
        player_a_prob, player_b_prob = no_vig_probabilities(1.80, 2.10)
        self.assertAlmostEqual(player_a_prob + player_b_prob, 1.0)
        self.assertGreater(player_a_prob, player_b_prob)

    def test_loader_filters_non_completed_and_orients_rows(self) -> None:
        rows = [
            ["2024-01-01", "Hard", "Beta", "Alpha", "1.40", "3.00", "Completed"],
            ["2024-01-02", "Hard", "Gamma", "Alpha", "1.50", "2.70", "Retired"],
        ]

        matches = self._load_rows(rows)

        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match.player_a, "Alpha")
        self.assertEqual(match.player_b, "Beta")
        self.assertFalse(match.player_a_won)
        self.assertAlmostEqual(match.player_a_odds, 3.00)
        self.assertAlmostEqual(match.player_b_odds, 1.40)

    def test_backtest_settles_bets_against_actual_outcome(self) -> None:
        rows = [["2024-01-01", "Hard", "Beta", "Alpha", "1.40", "3.00", "Completed"]]
        matches = self._load_rows(rows)

        result = run_backtest(matches, BacktestConfig(min_ev=0.0, odds_source="ps", model="elo"))

        self.assertEqual(result.summary.matches, 1)
        self.assertEqual(result.summary.bets, 1)
        self.assertEqual(result.predictions[0].bet_player, "Alpha")
        self.assertEqual(result.predictions[0].bet_side, "a")
        self.assertAlmostEqual(result.predictions[0].profit, -1.0)
        self.assertEqual(result.summary.hit_rate, 0.0)

    def test_backtest_runs_walk_forward(self) -> None:
        rows = [
            ["2024-01-01", "Hard", "Beta", "Alpha", "1.40", "3.00", "Completed"],
            ["2024-01-05", "Hard", "Beta", "Gamma", "1.70", "2.20", "Completed"],
        ]
        matches = self._load_rows(rows)
        result = run_backtest(matches, BacktestConfig(min_ev=0.0, odds_source="ps", model="elo"))

        self.assertEqual(result.summary.matches, 2)
        self.assertEqual(len(result.predictions), 2)
        self.assertGreater(result.predictions[1].model_player_a_prob, 0.5)

    def test_logreg_falls_back_cleanly_on_small_samples(self) -> None:
        rows = [
            ["2024-01-01", "Hard", "Alpha", "Beta", "1.60", "2.30", "Completed"],
            ["2024-01-02", "Hard", "Alpha", "Gamma", "1.70", "2.10", "Completed"],
            ["2024-01-03", "Hard", "Beta", "Gamma", "1.95", "1.85", "Completed"],
        ]
        matches = self._load_rows(rows)
        result = run_backtest(
            matches,
            BacktestConfig(
                min_ev=0.0,
                odds_source="ps",
                model="logreg",
                min_train_matches=1,
                retrain_interval=1,
            ),
        )

        self.assertEqual(result.summary.matches, 3)
        self.assertEqual(len(result.predictions), 3)
        self.assertIn(result.predictions[-1].probability_source, {"elo", "logreg"})

    def _load_rows(self, rows: list[list[str]]):
        TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with TEST_FILE.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(CSV_HEADER)
                writer.writerows(rows)

            return load_matches([TEST_FILE], odds_source="ps", completed_only=True)
        finally:
            if TEST_FILE.exists():
                TEST_FILE.unlink()


if __name__ == "__main__":
    unittest.main()
