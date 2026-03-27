from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class SegmentModelState:
    features: list[list[float]] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)
    estimator: Pipeline | None = None
    trained_examples: int = 0


class WalkForwardLogisticModel:
    def __init__(
        self,
        min_train_matches: int,
        train_window: int,
        retrain_interval: int,
    ) -> None:
        self.min_train_matches = min_train_matches
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self._segments: dict[str, SegmentModelState] = {}

    def predict(self, segment: str, features: list[float], fallback_probability: float) -> tuple[float, str]:
        state = self._segments.setdefault(segment, SegmentModelState())
        if len(state.outcomes) < self.min_train_matches:
            return fallback_probability, "elo"

        if state.estimator is None or len(state.outcomes) - state.trained_examples >= self.retrain_interval:
            state.estimator = self._fit_estimator(state)
            state.trained_examples = len(state.outcomes)

        if state.estimator is None:
            return fallback_probability, "elo"

        probability = float(state.estimator.predict_proba([features])[0, 1])
        return min(max(probability, 1e-6), 1.0 - 1e-6), "logreg"

    def update(self, segment: str, features: list[float], outcome: int) -> None:
        state = self._segments.setdefault(segment, SegmentModelState())
        state.features.append(features)
        state.outcomes.append(outcome)

    def _fit_estimator(self, state: SegmentModelState) -> Pipeline | None:
        if self.train_window > 0:
            features = state.features[-self.train_window :]
            outcomes = state.outcomes[-self.train_window :]
        else:
            features = state.features
            outcomes = state.outcomes

        if len(set(outcomes)) < 2:
            return None

        estimator = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=0.35,
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

        sample_weight = np.linspace(0.75, 1.25, num=len(outcomes), dtype=float)
        estimator.fit(features, outcomes, logreg__sample_weight=sample_weight)
        return estimator
