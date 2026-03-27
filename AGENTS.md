# AGENTS

## Purpose

This repository is a research scaffold for pre-match tennis betting models and walk-forward backtesting.
The current default path is:

- data ingestion from Tennis-Data `.xlsx` season files
- pre-match probability generation with `elo` or `logreg`
- per-tour adaptive decision filtering
- CSV export of per-match predictions

## Environment

- Python `>=3.11`
- Install dependencies with:

```powershell
python -m pip install -e .
```

## Canonical Commands

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Run the recommended backtest:

```powershell
python -m tennis_betting.cli --input data/raw/*.xlsx --model elo --odds-source ps --output data/processed/elo_adaptive_predictions.csv
```

Run the logistic model:

```powershell
python -m tennis_betting.cli --input data/raw/*.xlsx --model logreg --odds-source ps --output data/processed/logreg_adaptive_predictions.csv
```

## Repo Rules

- Treat all modeling and backtesting as strictly chronological. Do not introduce future leakage.
- Keep `data/raw/` and `data/processed/` out of commits except for `.gitkeep`.
- Do not commit `__pycache__`, temp folders, or machine-local artifacts.
- Prefer updating tests when changing loader, model, or settlement logic.
- Keep the CLI usable as the main entrypoint for research runs.

## Current Default Assumptions

- Default model: `elo`
- Default decision mode: `adaptive`
- Default odds source: `ps`
- Default minimum EV: `0.03`
- Adaptive tuning starts after `300` prior matches per tour and requires `40` historical bets to select a rule

## Next Priorities

- Add CLV tracking and use it in decision-layer evaluation
- Tune ATP and WTA separately instead of sharing one candidate grid
- Add tournament-level features from Tennis-Data fields such as `Series` or `Tier`
