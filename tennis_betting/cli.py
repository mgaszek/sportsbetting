from __future__ import annotations

import argparse
from pathlib import Path

from tennis_betting.backtest import run_backtest, write_predictions_csv
from tennis_betting.config import BacktestConfig, EloConfig
from tennis_betting.data import ODDS_PAIRS, expand_inputs, load_matches


ODDS_SOURCE_CHOICES = tuple(ODDS_PAIRS.keys())
MODEL_CHOICES = ("elo", "logreg")
DECISION_CHOICES = ("raw", "adaptive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tennis betting MVP backtest.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more CSV/XLSX paths or glob patterns.",
    )
    parser.add_argument("--min-ev", type=float, default=0.03, help="Base minimum EV to place a bet. 0.03 is the current best research default.")
    parser.add_argument("--stake", type=float, default=1.0, help="Flat stake per bet.")
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission rate applied to winning bets, e.g. 0.02 for 2%%.",
    )
    parser.add_argument(
        "--odds-source",
        choices=ODDS_SOURCE_CHOICES,
        default="ps",
        help="Odds source to use. 'ps' is the default executable baseline.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="elo",
        help="Prediction model to use. 'elo' is the current best betting baseline.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=DECISION_CHOICES,
        default="adaptive",
        help="Decision layer to use. 'adaptive' tunes conservative rules per tour from past matches only.",
    )
    parser.add_argument(
        "--decision-lookback",
        type=int,
        default=1500,
        help="Rolling history size for adaptive decision tuning.",
    )
    parser.add_argument(
        "--decision-min-history",
        type=int,
        default=300,
        help="Minimum past matches per tour before adaptive decision tuning starts.",
    )
    parser.add_argument(
        "--decision-retune-interval",
        type=int,
        default=100,
        help="How often to retune the adaptive decision layer, in matches per tour.",
    )
    parser.add_argument(
        "--decision-min-train-bets",
        type=int,
        default=40,
        help="Minimum historical bets required before a candidate decision rule can be selected.",
    )
    parser.add_argument(
        "--min-train-matches",
        type=int,
        default=400,
        help="Minimum past matches per tour before fitting the logistic model.",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=4000,
        help="Rolling training window size for the logistic model. Use 0 for all history.",
    )
    parser.add_argument(
        "--retrain-interval",
        type=int,
        default=200,
        help="Refit cadence for the logistic model in matches per tour.",
    )
    parser.add_argument("--k-factor", type=float, default=28.0, help="Elo K factor.")
    parser.add_argument(
        "--general-weight",
        type=float,
        default=0.35,
        help="Weight assigned to overall Elo in blended ratings.",
    )
    parser.add_argument(
        "--surface-weight",
        type=float,
        default=0.65,
        help="Weight assigned to surface Elo in blended ratings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to write per-match predictions to.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_paths = expand_inputs(args.input)
    if not input_paths:
        parser.error("No input files matched the provided paths or glob patterns.")

    if abs((args.general_weight + args.surface_weight) - 1.0) > 1e-9:
        parser.error("--general-weight and --surface-weight must sum to 1.0.")

    matches = load_matches(input_paths, odds_source=args.odds_source, completed_only=True)
    if not matches:
        parser.error("No matches were loaded. Check your CSV columns, odds source, and match comments.")

    config = BacktestConfig(
        min_ev=args.min_ev,
        stake=args.stake,
        commission=args.commission,
        odds_source=args.odds_source,
        model=args.model,
        min_train_matches=args.min_train_matches,
        train_window=args.train_window,
        retrain_interval=args.retrain_interval,
        decision_mode=args.decision_mode,
        decision_lookback=args.decision_lookback,
        decision_min_history=args.decision_min_history,
        decision_retune_interval=args.decision_retune_interval,
        decision_min_train_bets=args.decision_min_train_bets,
        elo=EloConfig(
            k_factor=args.k_factor,
            general_weight=args.general_weight,
            surface_weight=args.surface_weight,
        ),
    )
    result = run_backtest(matches, config)
    print_summary(result.summary, len(input_paths), args.odds_source, args.model, args.decision_mode)

    if args.output:
        write_predictions_csv(args.output, result.predictions)
        print(f"Predictions written to {args.output}")


def print_summary(summary, file_count: int, odds_source: str, model: str, decision_mode: str) -> None:
    print("Tennis Betting MVP Backtest")
    print(f"Files loaded:        {file_count}")
    print(f"Odds source:         {odds_source}")
    print(f"Model:               {model}")
    print(f"Decision mode:       {decision_mode}")
    print(f"Matches:             {summary.matches}")
    print(f"Bets placed:         {summary.bets}")
    print(f"Stake per bet:       {summary.stake_per_bet:.2f}")
    print(f"Turnover:            {summary.turnover:.2f}")
    print(f"Profit:              {summary.profit:.2f}")
    print(f"ROI:                 {summary.roi:.2%}")
    print(f"Hit rate:            {summary.hit_rate:.2%}")
    print(f"Average EV:          {summary.avg_expected_value:.2%}")
    print(f"Log loss:            {summary.log_loss:.4f}")
    print(f"Brier score:         {summary.brier_score:.4f}")


if __name__ == "__main__":
    main()
