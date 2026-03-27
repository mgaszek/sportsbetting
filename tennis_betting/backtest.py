from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from tennis_betting.config import BacktestConfig
from tennis_betting.data import MatchRecord
from tennis_betting.decision import AdaptiveDecisionLayer, Decision, DecisionRule
from tennis_betting.elo import EloRatings
from tennis_betting.ml import WalkForwardLogisticModel


def no_vig_probabilities(odds_a: float, odds_b: float) -> tuple[float, float]:
    implied_a = 1.0 / odds_a
    implied_b = 1.0 / odds_b
    total = implied_a + implied_b
    return implied_a / total, implied_b / total


def expected_value(probability: float, odds: float, commission: float) -> float:
    net_decimal = 1.0 + (odds - 1.0) * (1.0 - commission)
    return probability * net_decimal - 1.0


@dataclass(slots=True)
class Prediction:
    date: str
    tour: str
    surface: str
    tournament: str
    player_a: str
    player_b: str
    player_a_odds: float
    player_b_odds: float
    actual_winner: str
    outcome_a: int
    model_player_a_prob: float
    model_player_b_prob: float
    market_player_a_prob: float
    market_player_b_prob: float
    player_a_ev: float
    player_b_ev: float
    bet_side: str
    bet_player: str
    bet_odds: float
    expected_value: float
    profit: float
    rank_a: int | None
    rank_b: int | None
    points_a: int | None
    points_b: int | None
    odds_source: str
    probability_source: str
    decision_mode: str
    decision_market_blend: float
    decision_min_edge: float
    decision_min_ev: float
    decision_max_odds: float
    decision_min_prob: float
    source_file: str


@dataclass(slots=True)
class BacktestSummary:
    matches: int
    bets: int
    stake_per_bet: float
    turnover: float
    profit: float
    roi: float
    hit_rate: float
    avg_expected_value: float
    log_loss: float
    brier_score: float


@dataclass(slots=True)
class BacktestResult:
    summary: BacktestSummary
    predictions: list[Prediction]


def _clamp_probability(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _binary_log_loss(probability: float, outcome: int) -> float:
    probability = _clamp_probability(probability)
    return -(outcome * math.log(probability) + (1 - outcome) * math.log(1.0 - probability))


def _settle_profit(
    bet_side: str,
    match: MatchRecord,
    stake: float,
    commission: float,
) -> float:
    if bet_side == "a":
        if match.player_a_won:
            return (match.player_a_odds - 1.0) * stake * (1.0 - commission)
        return -stake
    if bet_side == "b":
        if not match.player_a_won:
            return (match.player_b_odds - 1.0) * stake * (1.0 - commission)
        return -stake
    return 0.0


def _raw_decision(
    model_prob_a: float,
    model_prob_b: float,
    odds_a: float,
    odds_b: float,
    config: BacktestConfig,
) -> Decision:
    rule = DecisionRule(
        market_blend=0.0,
        min_edge=0.0,
        min_ev=config.min_ev,
        max_odds=100.0,
        min_prob=0.0,
    )
    player_a_ev = expected_value(model_prob_a, odds_a, config.commission)
    player_b_ev = expected_value(model_prob_b, odds_b, config.commission)
    if player_a_ev >= config.min_ev and player_a_ev >= player_b_ev:
        return Decision(bet_side="a", bet_odds=odds_a, expected_value=player_a_ev, rule=rule)
    if player_b_ev >= config.min_ev:
        return Decision(bet_side="b", bet_odds=odds_b, expected_value=player_b_ev, rule=rule)
    return Decision(bet_side="", bet_odds=0.0, expected_value=0.0, rule=rule)


def run_backtest(matches: list[MatchRecord], config: BacktestConfig) -> BacktestResult:
    ratings = EloRatings(config.elo)
    predictor = None
    if config.model == "logreg":
        predictor = WalkForwardLogisticModel(
            min_train_matches=config.min_train_matches,
            train_window=config.train_window,
            retrain_interval=config.retrain_interval,
        )

    decision_layer = None
    if config.decision_mode == "adaptive":
        decision_layer = AdaptiveDecisionLayer(config)

    predictions: list[Prediction] = []
    total_log_loss = 0.0
    total_brier = 0.0

    for match in matches:
        elo_probability, feature_vector = ratings.feature_vector(match)
        model_player_a_prob = elo_probability
        probability_source = "elo"

        if predictor is not None:
            model_player_a_prob, probability_source = predictor.predict(
                segment=match.tour,
                features=feature_vector,
                fallback_probability=elo_probability,
            )

        model_player_a_prob = _clamp_probability(model_player_a_prob)
        model_player_b_prob = 1.0 - model_player_a_prob

        market_player_a_prob, market_player_b_prob = no_vig_probabilities(
            match.player_a_odds,
            match.player_b_odds,
        )

        player_a_ev = expected_value(
            model_player_a_prob,
            match.player_a_odds,
            config.commission,
        )
        player_b_ev = expected_value(
            model_player_b_prob,
            match.player_b_odds,
            config.commission,
        )

        if decision_layer is not None:
            decision = decision_layer.choose(
                segment=match.tour,
                model_prob_a=model_player_a_prob,
                market_prob_a=market_player_a_prob,
                odds_a=match.player_a_odds,
                odds_b=match.player_b_odds,
            )
        else:
            decision = _raw_decision(
                model_prob_a=model_player_a_prob,
                model_prob_b=model_player_b_prob,
                odds_a=match.player_a_odds,
                odds_b=match.player_b_odds,
                config=config,
            )

        bet_side = decision.bet_side
        bet_player = match.player_a if bet_side == "a" else match.player_b if bet_side == "b" else ""
        bet_odds = decision.bet_odds
        best_expected_value = decision.expected_value

        profit = _settle_profit(
            bet_side=bet_side,
            match=match,
            stake=config.stake,
            commission=config.commission,
        )

        predictions.append(
            Prediction(
                date=match.date.date().isoformat(),
                tour=match.tour,
                surface=match.surface,
                tournament=match.tournament,
                player_a=match.player_a,
                player_b=match.player_b,
                player_a_odds=match.player_a_odds,
                player_b_odds=match.player_b_odds,
                actual_winner=match.player_a if match.player_a_won else match.player_b,
                outcome_a=1 if match.player_a_won else 0,
                model_player_a_prob=model_player_a_prob,
                model_player_b_prob=model_player_b_prob,
                market_player_a_prob=market_player_a_prob,
                market_player_b_prob=market_player_b_prob,
                player_a_ev=player_a_ev,
                player_b_ev=player_b_ev,
                bet_side=bet_side,
                bet_player=bet_player,
                bet_odds=bet_odds,
                expected_value=best_expected_value,
                profit=profit,
                rank_a=match.rank_a,
                rank_b=match.rank_b,
                points_a=match.points_a,
                points_b=match.points_b,
                odds_source=match.odds_source,
                probability_source=probability_source,
                decision_mode=config.decision_mode,
                decision_market_blend=decision.rule.market_blend,
                decision_min_edge=decision.rule.min_edge,
                decision_min_ev=decision.rule.min_ev,
                decision_max_odds=decision.rule.max_odds,
                decision_min_prob=decision.rule.min_prob,
                source_file=match.source_file,
            )
        )

        outcome_a = 1 if match.player_a_won else 0
        total_log_loss += _binary_log_loss(model_player_a_prob, outcome_a)
        total_brier += (model_player_a_prob - outcome_a) ** 2

        if predictor is not None:
            predictor.update(match.tour, feature_vector, outcome_a)
        if decision_layer is not None:
            decision_layer.update(
                segment=match.tour,
                model_prob_a=model_player_a_prob,
                market_prob_a=market_player_a_prob,
                odds_a=match.player_a_odds,
                odds_b=match.player_b_odds,
                outcome_a=outcome_a,
            )
        ratings.update_match(match)

    summary = summarize(predictions, config, total_log_loss, total_brier)
    return BacktestResult(summary=summary, predictions=predictions)


def summarize(
    predictions: list[Prediction],
    config: BacktestConfig,
    total_log_loss: float,
    total_brier: float,
) -> BacktestSummary:
    bets = [prediction for prediction in predictions if prediction.bet_side]
    turnover = len(bets) * config.stake
    profit = sum(prediction.profit for prediction in bets)
    wins = sum(1 for prediction in bets if prediction.profit > 0)
    avg_ev = (
        sum(prediction.expected_value for prediction in bets) / len(bets) if bets else 0.0
    )

    return BacktestSummary(
        matches=len(predictions),
        bets=len(bets),
        stake_per_bet=config.stake,
        turnover=turnover,
        profit=profit,
        roi=(profit / turnover) if turnover else 0.0,
        hit_rate=(wins / len(bets)) if bets else 0.0,
        avg_expected_value=avg_ev,
        log_loss=(total_log_loss / len(predictions)) if predictions else 0.0,
        brier_score=(total_brier / len(predictions)) if predictions else 0.0,
    )


def write_predictions_csv(path: str | Path, predictions: list[Prediction]) -> None:
    if not predictions:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(predictions[0]).keys()))
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(asdict(prediction))
