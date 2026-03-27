# Tennis Betting MVP

This repository is a concrete starting point for a UK-compatible tennis betting research tool.

The project now supports two pre-match models plus an adaptive decision layer:

- `elo`: surface-aware Elo baseline
- `logreg`: walk-forward logistic regression trained on pre-match features generated from the historical stream
- `adaptive` bet filter: per-tour rule tuning that blends toward the market and tightens bet selection using past matches only

## Why tennis first

Tennis is a good first market because:

- UK-facing books and exchanges carry it heavily
- the target is binary in most pre-match markets
- player-vs-player structure is simpler than team sports
- historical match results and odds are easy to obtain

Current source options discussed for this MVP:

- Historical results and fixed odds: `https://www.tennis-data.co.uk/data.php`
- Live or current bookmaker odds in the UK region: `https://the-odds-api.com/liveapi/guides/v4/`
- Exchange odds and execution path: `https://www.betfair.com/exchange/plus/en/tennis-betting-2`

## Project layout

```text
data/
  raw/          # downloaded ATP/WTA season files
  processed/    # prediction exports and ad hoc analysis output
examples/
  sample_tennis_data.csv
tennis_betting/
  cli.py
  config.py
  data.py
  elo.py
  ml.py
  decision.py
  backtest.py
tests/
  test_backtest.py
```

## Input data shape

The loader supports tennis-data style CSVs and current Tennis-Data `.xlsx` season files.
It looks for these columns:

- `Date`
- `Surface`
- `Winner`
- `Loser`
- optional pre-match fields such as `WRank`, `LRank`, `WPts`, `LPts`, `Court`, `Round`, `Best of`, `Comment`
- an odds pair from one of:
  - `PSW` / `PSL`
  - `B365W` / `B365L`
  - `AvgW` / `AvgL`
  - `MaxW` / `MaxL`

Rows with `Comment` other than blank or `Completed` are excluded by default.
Legacy `.xls` files are not supported yet.

## Models

### Elo baseline

The Elo baseline keeps separate player states by tour and tracks:

- overall Elo
- surface-specific Elo
- context-specific ratings for surface plus indoor/outdoor court
- recent form and recent same-surface form
- matches played and days since last match

The baseline probability still comes from blended overall plus surface Elo.

### Walk-forward logistic model

The logistic model uses only pre-match features available at prediction time and trains separately by tour in strict chronological order.
It uses:

- Elo logit and rating gaps
- surface/context rating gaps
- recent form and same-surface form
- inactivity features
- matches-played differential
- rank and points differentials when available
- surface flags and best-of-five / indoor indicators

The model refits on a rolling window and only begins predicting after a minimum amount of past tour data has accumulated.
Before that, it falls back to the Elo baseline.

## Bet rule

The default decision layer is now `adaptive`. For each match it can:

- blend model probability back toward the no-vig market probability
- require a minimum edge over the market
- cap acceptable odds to avoid low-quality longshots
- require a minimum EV and probability threshold before betting

All candidate rules are scored only on prior matches from the same tour. If you want the old behavior, run `--decision-mode raw`.

Expected value per 1 unit stake with commission on winnings:

```text
EV = p * (1 + (odds - 1) * (1 - commission)) - 1
```

## Quick start

Run the recommended Elo plus adaptive decision layer:

```powershell
python -m tennis_betting.cli --input data/raw/*.xlsx --model elo --odds-source ps
```

Run the walk-forward logistic model with the same decision layer:

```powershell
python -m tennis_betting.cli --input data/raw/*.xlsx --model logreg --odds-source ps --output data/processed/logreg_adaptive_predictions.csv
```

## Current baseline snapshot

Using the downloaded `2024-2026` ATP/WTA season files in `data/raw/` and Pinnacle odds:

- raw `elo`: `0.6482` log loss, `0.2287` Brier, `-7.77%` ROI
- adaptive `elo` with current defaults: `0.6482` log loss, `0.2287` Brier, `-3.34%` ROI
- adaptive `logreg` with current defaults: `0.6403` log loss, `0.2244` Brier, `-4.86%` ROI

So the decision layer is materially better than the raw EV rule, but the system is still not profitable. Elo remains the better betting baseline, while logistic regression still has the better probability metrics.

## Practical next steps

1. Add explicit ATP/WTA-specific hyperparameter tuning instead of sharing defaults.
2. Add market-line movement or closing line value tracking.
3. Add tournament-level features from `Series` / `Tier`.
4. Add CLV tracking and use it alongside realized ROI when ranking decision rules.
5. Add news or injury features only after the structured model is stable.

## Notes

- This scaffold is for research and system development, not betting advice.
- The default example uses fixed 1-unit stakes.
- Exchange commission, rejected bets, and liquidity constraints are only modeled at a simple placeholder level.
