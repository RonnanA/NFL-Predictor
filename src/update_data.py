import pandas as pd
from pathlib import Path

from config import YEAR, PRODUCTION_DIR, TEST_DIR, RAW_DIR
from scraper import scrape_season, scrape_stats
from reformater import reformat_season
from merge_files import merge, get_master_df
from features import data_cleanup, build_team_snapshots, build_matchup_frame
from model import rolling_cross_validate, train_final, load_model


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def refresh_data():
    print("scraping season...", flush=True)
    df = scrape_season(YEAR)
    
    if df is not None and not df.empty:
        print("reformatting season...", flush=True)
        reformat_season(df, YEAR)
    else:
        print(f"season file already up to date or scrape failed, skipping reformat", flush=True)

    season_path = RAW_DIR / f"Season-{YEAR}.csv"
    if not season_path.exists():
        print(f"season file missing for {YEAR} — skipping stats scrape. retry scraping manually.", flush=True)
        return

    print("scraping stats...", flush=True)
    scrape_stats(YEAR)

    print("merging files...", flush=True)
    merge(YEAR)

    print("data refresh complete", flush=True)


def load_master(years: list[int] | None = None) -> tuple:
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024, 2025]

    season_dfs = []
    for year in years:
        raw = get_master_df(year)
        if raw is None or raw.empty:
            continue
        cleaned = data_cleanup(raw, season=year)
        season_dfs.append(cleaned)

    if not season_dfs:
        raise RuntimeError("no season data loaded — check your processed CSV files")

    all_games = pd.concat(season_dfs).sort_values(["season", "week"]).reset_index(drop=True)

    snapshots_df = build_team_snapshots(all_games)
    matchup_df   = build_matchup_frame(all_games, snapshots_df)

    return snapshots_df, matchup_df


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def get_or_train_model(matchup_df, *, model_path: Path, force_retrain: bool = False) -> object:
    if model_path.exists() and not force_retrain:
        print(f"loading model from {model_path}")
        return load_model(model_path)

    print("training new model...")
    return train_final(matchup_df, save_path=model_path)


# ─────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────

def setup_for_predict(update_data: bool = False, update_model: bool = False) -> tuple:
    if update_data:
        refresh_data()

    snapshots_df, matchup_df = load_master()
    pipe = get_or_train_model(
        matchup_df,
        model_path=PRODUCTION_DIR / "nfl_rf_model.pkl",
        force_retrain=update_model,
    )
    return pipe, snapshots_df


def setup_for_eval(update_data: bool = False, update_model: bool = False) -> tuple:
    if update_data:
        refresh_data()

    snapshots_df, matchup_df = load_master()

    cv_results = rolling_cross_validate(matchup_df)

    pipe = get_or_train_model(
        matchup_df,
        model_path=TEST_DIR / "test_nfl_rf_model.pkl",
        force_retrain=update_model,
    )
    return pipe, snapshots_df, cv_results