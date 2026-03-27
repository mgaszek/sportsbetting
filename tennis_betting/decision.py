from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from tennis_betting.config import BacktestConfig


def _expected_value(probability: float, odds: float, commission: float) -> float:
    net_decimal = 1.0 + (odds - 1.0) * (1.0 - commission)
    return probability * net_decimal - 1.0


@dataclass(slots=True)
class DecisionRule:
    market_blend: float
    min_edge: float
    min_ev: float
    max_odds: float
    min_prob: float


@dataclass(slots=True)
class DecisionExample:
    model_prob_a: float
    market_prob_a: float
    odds_a: float
    odds_b: float
    outcome_a: int


@dataclass(slots=True)
class DecisionSegmentState:
    history: list[DecisionExample] = field(default_factory=list)
    current_rule: DecisionRule | None = None
    last_tuned_size: int = 0


@dataclass(slots=True)
class Decision:
    bet_side: str
    bet_odds: float
    expected_value: float
    rule: DecisionRule


class AdaptiveDecisionLayer:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.default_rule = DecisionRule(
            market_blend=0.0,
            min_edge=0.0,
            min_ev=config.min_ev,
            max_odds=100.0,
            min_prob=0.0,
        )
        min_ev_grid = sorted(
            {
                config.min_ev,
                max(config.min_ev, 0.03),
                max(config.min_ev, 0.05),
                max(config.min_ev, 0.07),
            }
        )
        self.candidates = [
            DecisionRule(*values)
            for values in product(
                [0.0, 0.3, 0.4, 0.5, 0.6],
                [0.0, 0.005, 0.01, 0.02, 0.03],
                min_ev_grid,
                [1.8, 2.0, 2.5, 3.0],
                [0.0, 0.5, 0.55, 0.6],
            )
        ]
        self._segments: dict[str, DecisionSegmentState] = {}

    def choose(
        self,
        segment: str,
        model_prob_a: float,
        market_prob_a: float,
        odds_a: float,
        odds_b: float,
    ) -> Decision:
        state = self._segments.setdefault(segment, DecisionSegmentState(current_rule=self.default_rule))
        self._maybe_retune(state)
        return self._apply_rule(
            state.current_rule or self.default_rule,
            model_prob_a=model_prob_a,
            market_prob_a=market_prob_a,
            odds_a=odds_a,
            odds_b=odds_b,
            commission=self.config.commission,
        )

    def update(
        self,
        segment: str,
        model_prob_a: float,
        market_prob_a: float,
        odds_a: float,
        odds_b: float,
        outcome_a: int,
    ) -> None:
        state = self._segments.setdefault(segment, DecisionSegmentState(current_rule=self.default_rule))
        state.history.append(
            DecisionExample(
                model_prob_a=model_prob_a,
                market_prob_a=market_prob_a,
                odds_a=odds_a,
                odds_b=odds_b,
                outcome_a=outcome_a,
            )
        )

    def _maybe_retune(self, state: DecisionSegmentState) -> None:
        history_size = len(state.history)
        if history_size < self.config.decision_min_history:
            return
        if state.current_rule is None:
            state.current_rule = self.default_rule
        if history_size == self.config.decision_min_history or history_size - state.last_tuned_size >= self.config.decision_retune_interval:
            state.current_rule = self._best_rule(state.history[-self.config.decision_lookback :])
            state.last_tuned_size = history_size

    def _best_rule(self, history: list[DecisionExample]) -> DecisionRule:
        model_prob_a = np.array([row.model_prob_a for row in history], dtype=float)
        market_prob_a = np.array([row.market_prob_a for row in history], dtype=float)
        odds_a = np.array([row.odds_a for row in history], dtype=float)
        odds_b = np.array([row.odds_b for row in history], dtype=float)
        outcome_a = np.array([row.outcome_a for row in history], dtype=int)

        best_rule = self.default_rule
        best_score = float("-inf")

        for rule in self.candidates:
            blend_prob_a = (1.0 - rule.market_blend) * model_prob_a + rule.market_blend * market_prob_a
            blend_prob_b = 1.0 - blend_prob_a
            market_prob_b = 1.0 - market_prob_a

            edge_a = blend_prob_a - market_prob_a
            edge_b = blend_prob_b - market_prob_b

            ev_a = np.array(
                [_expected_value(prob, odds, self.config.commission) for prob, odds in zip(blend_prob_a, odds_a)],
                dtype=float,
            )
            ev_b = np.array(
                [_expected_value(prob, odds, self.config.commission) for prob, odds in zip(blend_prob_b, odds_b)],
                dtype=float,
            )

            choose_a = (
                (odds_a <= rule.max_odds)
                & (blend_prob_a >= rule.min_prob)
                & (edge_a >= rule.min_edge)
                & (ev_a >= rule.min_ev)
                & (ev_a >= ev_b)
            )
            choose_b = (
                (odds_b <= rule.max_odds)
                & (blend_prob_b >= rule.min_prob)
                & (edge_b >= rule.min_edge)
                & (ev_b >= rule.min_ev)
                & (~choose_a)
            )
            bet_count = int(choose_a.sum() + choose_b.sum())
            if bet_count < self.config.decision_min_train_bets:
                continue

            net_win_a = (odds_a - 1.0) * (1.0 - self.config.commission)
            net_win_b = (odds_b - 1.0) * (1.0 - self.config.commission)
            profit_a = np.where(outcome_a == 1, net_win_a, -1.0)
            profit_b = np.where(outcome_a == 0, net_win_b, -1.0)
            profit = float(profit_a[choose_a].sum() + profit_b[choose_b].sum())
            roi = profit / bet_count
            score = profit + 15.0 * roi
            if score > best_score:
                best_score = score
                best_rule = rule

        return best_rule

    def _apply_rule(
        self,
        rule: DecisionRule,
        model_prob_a: float,
        market_prob_a: float,
        odds_a: float,
        odds_b: float,
        commission: float,
    ) -> Decision:
        blend_prob_a = (1.0 - rule.market_blend) * model_prob_a + rule.market_blend * market_prob_a
        blend_prob_b = 1.0 - blend_prob_a
        market_prob_b = 1.0 - market_prob_a
        edge_a = blend_prob_a - market_prob_a
        edge_b = blend_prob_b - market_prob_b
        ev_a = _expected_value(blend_prob_a, odds_a, commission)
        ev_b = _expected_value(blend_prob_b, odds_b, commission)

        if (
            odds_a <= rule.max_odds
            and blend_prob_a >= rule.min_prob
            and edge_a >= rule.min_edge
            and ev_a >= rule.min_ev
            and ev_a >= ev_b
        ):
            return Decision(bet_side="a", bet_odds=odds_a, expected_value=ev_a, rule=rule)

        if (
            odds_b <= rule.max_odds
            and blend_prob_b >= rule.min_prob
            and edge_b >= rule.min_edge
            and ev_b >= rule.min_ev
        ):
            return Decision(bet_side="b", bet_odds=odds_b, expected_value=ev_b, rule=rule)

        return Decision(bet_side="", bet_odds=0.0, expected_value=0.0, rule=rule)
