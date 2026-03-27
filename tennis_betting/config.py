from dataclasses import dataclass, field


@dataclass(slots=True)
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 28.0
    general_weight: float = 0.35
    surface_weight: float = 0.65


@dataclass(slots=True)
class BacktestConfig:
    min_ev: float = 0.03
    stake: float = 1.0
    commission: float = 0.0
    odds_source: str = "ps"
    model: str = "elo"
    min_train_matches: int = 400
    train_window: int = 4000
    retrain_interval: int = 200
    decision_mode: str = "adaptive"
    decision_lookback: int = 1500
    decision_min_history: int = 300
    decision_retune_interval: int = 100
    decision_min_train_bets: int = 40
    elo: EloConfig = field(default_factory=EloConfig)
