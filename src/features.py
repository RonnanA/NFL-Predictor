import pandas as pd
import numpy as np

ROLLING_WINDOW = 5
REST_DAYS_DEFAULT = 7  # assigned to week 1 since there's no prior game

POSTSEASON_LABELS = [
    "WildCard", "Wildcard", "Divisional", "Division",
    "ConfChamp", "ConferenceChampionship", "SuperBowl"
]

DROP_COLS = [
    "url", "week_day", "game_time",
    "home_time_of_possession", "away_time_of_possession"
]

STAT_COLS = [
    "score", "rush_yds", "pass_yds", "total_yards",
    "first_downs", "turnovers", "pen_yds"
]


# ─────────────────────────────────────────────
# Raw cleanup
# ─────────────────────────────────────────────

def data_cleanup(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = df.copy()
    df["season"] = season
    df = df[~df["week"].isin(POSTSEASON_LABELS)]

    non_numeric = {"home_team", "away_team", "event_date"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    df = (
        df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        .dropna(subset=["week", "home_score", "away_score"])
        .sort_values(["season", "week"])
        .reset_index(drop=True)
    )
    return df


# ─────────────────────────────────────────────
# Build per-team snapshots
# ─────────────────────────────────────────────

def build_team_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    home_view = _extract_team_view(df, perspective="home")
    away_view = _extract_team_view(df, perspective="away")
    long = pd.concat([home_view, away_view]).sort_values(["season", "week"]).reset_index(drop=True)

    long = _add_rest_days(long)
    long = _add_rolling_avgs(long)
    long = _add_season_to_date_avgs(long)
    long = _add_streak(long)

    # drop event_date now that rest_days is computed
    long = long.drop(columns=["event_date"] + STAT_COLS + ["won"])

    return long.reset_index(drop=True)


def _extract_team_view(df: pd.DataFrame, perspective: str) -> pd.DataFrame:
    p = perspective  # "home" or "away"
    opp = "away" if p == "home" else "home"

    view = pd.DataFrame({
        "season":     df["season"],
        "week":       df["week"],
        "event_date": df["event_date"],
        "team":       df[f"{p}_team"],
        "score":      df[f"{p}_score"],
        "opp_score":  df[f"{opp}_score"],
        "rush_yds":   df[f"{p}_rush_yds"],
        "pass_yds":   df[f"{p}_pass_yds"],
        "total_yards":df[f"{p}_total_yards"],
        "first_downs":df[f"{p}_first_downs"],
        "turnovers":  df[f"{p}_turnovers"],
        "pen_yds":    df[f"{p}_pen_yds"],
        "won":        (df[f"{p}_score"] > df[f"{opp}_score"]).astype(int),
    })
    return view


def _add_rest_days(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["team", "season", "week"]).copy()
    long["prev_date"] = long.groupby("team")["event_date"].shift(1)
    long["rest_days"] = (long["event_date"] - long["prev_date"]).dt.days
    long["rest_days"] = long["rest_days"].fillna(REST_DAYS_DEFAULT)
    long = long.drop(columns=["prev_date"])
    return long


def _add_rolling_avgs(long: pd.DataFrame) -> pd.DataFrame:
    for stat in STAT_COLS:
        long[f"roll_{stat}"] = (
            long.groupby("team")[stat]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )
    return long


def _add_season_to_date_avgs(long: pd.DataFrame) -> pd.DataFrame:
    for stat in STAT_COLS:
        long[f"std_{stat}"] = (
            long.groupby(["team", "season"])[stat]
            .transform(lambda s: s.shift(1).expanding().mean())
        )
    return long


def _add_streak(long: pd.DataFrame) -> pd.DataFrame:
    def compute_streak(won_series: pd.Series) -> pd.Series:
        streaks = []
        current = 0
        for i, won in enumerate(won_series):
            if i == 0:
                streaks.append(0)  # no prior game
                current = 1 if won else -1
                continue
            streaks.append(current)
            if won:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
        return pd.Series(streaks, index=won_series.index)

    long["streak"] = (
        long.groupby(["team", "season"])["won"]
        .transform(compute_streak)
    )
    return long


# ─────────────────────────────────────────────
# Build matchup frame for training/prediction
# ─────────────────────────────────────────────

def build_matchup_frame(games_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = (
        [f"roll_{s}" for s in STAT_COLS] +
        [f"std_{s}"  for s in STAT_COLS] +
        ["streak", "rest_days"]
    )

    home_snaps = _lookup_prior_snapshot(games_df, snapshots_df, perspective="home")
    away_snaps = _lookup_prior_snapshot(games_df, snapshots_df, perspective="away")

    matchups = games_df[["season", "week", "home_team", "away_team"]].copy()
    matchups["home_team_wins"] = (games_df["home_score"] > games_df["away_score"]).astype(int)

    stat_features = [c for c in feature_cols if c != "rest_days"]
    for col in stat_features:
        matchups[f"diff_{col}"] = home_snaps[col].values - away_snaps[col].values

    # rest days are not diffed — both are meaningful signals independently
    matchups["home_rest_days"] = home_snaps["rest_days"].values
    matchups["away_rest_days"] = away_snaps["rest_days"].values

    return matchups.dropna().reset_index(drop=True)


def _lookup_prior_snapshot(
    games_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    perspective: str
) -> pd.DataFrame:
    team_col = f"{perspective}_team"
    results = []

    for _, game in games_df.iterrows():
        team   = game[team_col]
        season = game["season"]
        week   = game["week"]

        # look for prior games this season
        prior = snapshots_df[
            (snapshots_df["team"] == team) &
            (snapshots_df["season"] == season) &
            (snapshots_df["week"] < week)
        ]

        # fall back to previous season if no prior games yet (week 1)
        if prior.empty:
            prior = snapshots_df[
                (snapshots_df["team"] == team) &
                (snapshots_df["season"] == season - 1)
            ]

        if prior.empty:
            results.append(None)
        else:
            results.append(prior.sort_values("week").iloc[-1])

    valid = [r for r in results if r is not None]
    if len(valid) != len(results):
        # rows with no snapshot will be dropped by dropna() in build_matchup_frame
        placeholder = pd.Series({c: np.nan for c in valid[0].index})
        results = [r if r is not None else placeholder for r in results]

    return pd.DataFrame(results).reset_index(drop=True)