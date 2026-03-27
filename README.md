# NFL Game Predictor
> **This project is retired.** The data source (Pro Football Reference) has implemented Cloudflare bot protection that reliably blocks automated scraping. The project is preserved as a portfolio piece demonstrating an end-to-end ML pipeline.

---

NFL Game Predictor. A data-driven machine learning project that predicts the outcome of an NFL matchup using team performance statistics from previous seasons.

It scrapes, cleans, and analyzes historical game data, then trains a predictive model that outputs the probability of a home team winning any given matchup.

## Project Overview
This project answers the question:
> *Given two NFL teams, what are the chances the home team wins?*

It uses a combination of:
- Web scraping (automated data collection from Pro Football Reference via Playwright)
- Data processing and feature engineering
- Machine learning with scikit-learn (RandomForestClassifier with probability calibration)

## How It Works

### Data Collection
The scraper collects from Pro Football Reference:
- Every regular-season game result for any modern-era season
- Detailed per-game team stats (passing/rushing yards, turnovers, penalties, etc.)

Data is saved as two raw files per season, season schedule and per-game stats, which are merged into a processed CSV.

> **Note:** Pro Football Reference added Cloudflare Turnstile bot protection in early 2026, which blocks headless browsers and HTTP clients regardless of impersonation strategy. Re-scraping data on a new machine is no longer reliably possible without a manual workaround.

### Data Cleanup
Raw data is cleaned before training. Postseason games, metadata columns, and rows with missing scores are removed. `event_date` is preserved through the pipeline for rest day calculations before being dropped.

### Feature Engineering
The pipeline builds a **team snapshot store**, one row per team per game, representing each team's current state before that game. Features are computed strictly from prior games only (no lookahead).

Features per team include:

- **Rolling averages** (5-game window) - recent form across score, rush yards, pass yards, total yards, first downs, turnovers, and penalty yards
- **Season-to-date averages** - overall strength within the current season, resets each season
- **Win/loss streak** - signed integer representing consecutive wins (+) or losses (-), resets each season
- **Rest days** - days since the team's last game, week 1 defaults to 7

Matchup features are computed as home-minus-away diffs for all rolling and season-to-date stats, with rest days kept as independent columns for each team.

### Model Training
The model uses a scikit-learn pipeline:
- `StandardScaler` for feature normalization
- `RandomForestClassifier` wrapped in `CalibratedClassifierCV` to produce well-calibrated win probabilities rather than raw vote fractions

Training uses **time-based rolling window cross-validation**, each fold trains on all seasons up to N and tests on season N+1, which gives an honest estimate of how the model generalizes forward in time. Metrics reported per fold: accuracy, ROC AUC, and Brier score.

### Making Predictions
Once trained, the model predicts the probability of a home team winning a future matchup using each team's most recent snapshot from the feature store.

Example output:
```
BUF @ KC
predicted winner:  KC
home win prob:     61.3%
confidence:        61.3%
```

## Why Retired
Pro Football Reference implemented Cloudflare Turnstile verification in early 2026. This challenge requires real browser interaction to pass and cannot be reliably bypassed by headless browsers (Playwright) or TLS-impersonating HTTP clients (curl_cffi). Since the entire data collection layer depends on PFR, the project cannot be run end-to-end on a new machine without manually downloading data files from a real browser session.

The ML pipeline (`features.py`, `model.py`, `predict.py`, `update_data.py`) is complete, tested, and functional given existing processed data.