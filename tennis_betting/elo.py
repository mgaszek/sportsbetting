from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from tennis_betting.config import EloConfig
from tennis_betting.data import MatchRecord


SURFACE_NAMES = ("Hard", "Clay", "Grass", "Carpet")
DEFAULT_INACTIVITY_DAYS = 21.0
RECENT_FORM_WINDOW = 10


def win_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass(slots=True)
class PlayerState:
    overall: float
    by_surface: dict[str, float] = field(default_factory=dict)
    by_context: dict[str, float] = field(default_factory=dict)
    matches_played: int = 0
    last_played: datetime | None = None
    recent_results: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_FORM_WINDOW)
    )
    recent_surface_results: dict[str, deque[int]] = field(default_factory=dict)


class EloRatings:
    def __init__(self, config: EloConfig) -> None:
        self.config = config
        self._players: dict[str, PlayerState] = {}

    def _player_key(self, tour: str, name: str) -> str:
        return f"{tour}:{name}"

    def _player(self, key: str) -> PlayerState:
        if key not in self._players:
            self._players[key] = PlayerState(overall=self.config.base_rating)
        return self._players[key]

    def _context_key(self, surface: str, court: str) -> str:
        court_label = "indoor" if "indoor" in court.casefold() else "outdoor"
        return f"{surface}|{court_label}"

    def _surface_rating(self, state: PlayerState, surface: str) -> float:
        return state.by_surface.get(surface, self.config.base_rating)

    def _context_rating(self, state: PlayerState, surface: str, court: str) -> float:
        return state.by_context.get(self._context_key(surface, court), self.config.base_rating)

    def _recent_win_rate(self, history: deque[int] | None) -> float:
        if not history:
            return 0.5
        return sum(history) / len(history)

    def _days_since_last(self, state: PlayerState, match_date: datetime) -> float:
        if state.last_played is None:
            return DEFAULT_INACTIVITY_DAYS
        return float(max((match_date.date() - state.last_played.date()).days, 0))

    def _rank_advantage(self, rank_a: int | None, rank_b: int | None) -> tuple[float, float]:
        if rank_a is None or rank_b is None:
            return 0.0, 0.0
        return math.log1p(rank_b) - math.log1p(rank_a), 1.0

    def _points_difference(self, points_a: int | None, points_b: int | None) -> tuple[float, float]:
        if points_a is None or points_b is None:
            return 0.0, 0.0
        return math.log1p(points_a) - math.log1p(points_b), 1.0

    def blended_rating(self, tour: str, player: str, surface: str) -> float:
        state = self._player(self._player_key(tour, player))
        surface_rating = self._surface_rating(state, surface)
        return (
            state.overall * self.config.general_weight
            + surface_rating * self.config.surface_weight
        )

    def predict(self, tour: str, player_a: str, player_b: str, surface: str) -> float:
        return win_probability(
            self.blended_rating(tour, player_a, surface),
            self.blended_rating(tour, player_b, surface),
        )

    def feature_vector(self, match: MatchRecord) -> tuple[float, list[float]]:
        player_a_state = self._player(self._player_key(match.tour, match.player_a))
        player_b_state = self._player(self._player_key(match.tour, match.player_b))

        overall_diff = (player_a_state.overall - player_b_state.overall) / 400.0
        surface_diff = (
            self._surface_rating(player_a_state, match.surface)
            - self._surface_rating(player_b_state, match.surface)
        ) / 400.0
        context_diff = (
            self._context_rating(player_a_state, match.surface, match.court)
            - self._context_rating(player_b_state, match.surface, match.court)
        ) / 400.0

        elo_probability = self.predict(match.tour, match.player_a, match.player_b, match.surface)
        elo_probability = min(max(elo_probability, 1e-6), 1.0 - 1e-6)
        elo_logit = math.log(elo_probability / (1.0 - elo_probability))

        player_a_surface_history = player_a_state.recent_surface_results.get(match.surface)
        player_b_surface_history = player_b_state.recent_surface_results.get(match.surface)
        recent_form_diff = self._recent_win_rate(player_a_state.recent_results) - self._recent_win_rate(
            player_b_state.recent_results
        )
        recent_surface_form_diff = self._recent_win_rate(player_a_surface_history) - self._recent_win_rate(
            player_b_surface_history
        )

        days_since_last_a = min(self._days_since_last(player_a_state, match.date), 120.0)
        days_since_last_b = min(self._days_since_last(player_b_state, match.date), 120.0)
        matches_diff = math.log1p(player_a_state.matches_played) - math.log1p(player_b_state.matches_played)
        rank_advantage, rank_known = self._rank_advantage(match.rank_a, match.rank_b)
        points_diff, points_known = self._points_difference(match.points_a, match.points_b)
        indoor = 1.0 if "indoor" in match.court.casefold() else 0.0
        best_of_five = 1.0 if (match.best_of or 0) >= 5 else 0.0

        surface_flags = [1.0 if match.surface == surface_name else 0.0 for surface_name in SURFACE_NAMES]

        features = [
            elo_logit,
            overall_diff,
            surface_diff,
            context_diff,
            recent_form_diff,
            recent_surface_form_diff,
            matches_diff,
            days_since_last_a / 30.0,
            days_since_last_b / 30.0,
            (days_since_last_b - days_since_last_a) / 30.0,
            rank_advantage,
            rank_known,
            points_diff,
            points_known,
            indoor,
            best_of_five,
            *surface_flags,
        ]
        return elo_probability, features

    def update_match(self, match: MatchRecord) -> None:
        player_a_key = self._player_key(match.tour, match.player_a)
        player_b_key = self._player_key(match.tour, match.player_b)
        player_a_state = self._player(player_a_key)
        player_b_state = self._player(player_b_key)

        winner_key = player_a_key if match.player_a_won else player_b_key
        loser_key = player_b_key if match.player_a_won else player_a_key
        winner_state = self._player(winner_key)
        loser_state = self._player(loser_key)

        expected_winner = self.predict(
            match.tour,
            match.player_a if match.player_a_won else match.player_b,
            match.player_b if match.player_a_won else match.player_a,
            match.surface,
        )
        k_winner = self._effective_k(winner_state)
        k_loser = self._effective_k(loser_state)
        delta = ((k_winner + k_loser) / 2.0) * (1.0 - expected_winner)

        winner_state.overall += delta
        loser_state.overall -= delta

        winner_state.by_surface[match.surface] = self._surface_rating(winner_state, match.surface) + delta
        loser_state.by_surface[match.surface] = self._surface_rating(loser_state, match.surface) - delta

        context_key = self._context_key(match.surface, match.court)
        winner_state.by_context[context_key] = winner_state.by_context.get(context_key, self.config.base_rating) + delta
        loser_state.by_context[context_key] = loser_state.by_context.get(context_key, self.config.base_rating) - delta

        self._record_result(player_a_state, match.surface, match.player_a_won, match.date)
        self._record_result(player_b_state, match.surface, not match.player_a_won, match.date)

    def _effective_k(self, state: PlayerState) -> float:
        if state.matches_played < 20:
            return self.config.k_factor * 1.35
        if state.matches_played < 50:
            return self.config.k_factor * 1.15
        return self.config.k_factor

    def _record_result(
        self,
        state: PlayerState,
        surface: str,
        won: bool,
        match_date: datetime,
    ) -> None:
        result = 1 if won else 0
        state.matches_played += 1
        state.last_played = match_date
        state.recent_results.append(result)
        state.recent_surface_results.setdefault(surface, deque(maxlen=RECENT_FORM_WINDOW)).append(result)
