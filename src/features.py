import pandas as pd


def data_cleanup(df: pd.DataFrame, season: int) -> pd.DataFrame:
    
    df["season"] = season

    postseason_rows = ["WildCard", "Wildcard", "Divisional", "Division", "ConfChamp", "ConferenceChampionship", "SuperBowl"]
    df = df[~df['week'].isin(postseason_rows)]

    drop_cols = ["url", "week_day", "event_date", "game_time", "home_time_of_possession", "away_time_of_possession"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    non_numeric = ["week", "home_team", "away_team", "season"]
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["week"]).sort_values("week").reset_index(drop=True)

    return df


def add_team_rolling_averages(df: pd.DataFrame, based=3):
    df = df.sort_values(["season", "week"]).copy()

    stats = [
        "score", "first_downs", "rush_yds", "pass_yds",
        "total_yards", "turnovers", "pen_yds"
    ]

    for stat in stats:
        df[f"home_{stat}_avg"] = (
            df.groupby("home_team")[f"home_{stat}"]
            .shift(1)
            .rolling(based)
            .mean()
        )
        df[f"away_{stat}_avg"] = (
            df.groupby("away_team")[f"away_{stat}"]
            .shift(1)
            .rolling(based)
            .mean()
        )

    df = df.dropna(subset=[c for c in df.columns if c.endswith("_avg")])

    return df


def build_features(df: pd.DataFrame, based=3):
    df = add_team_rolling_averages(df, based=based)

    diff_stats = [
        "score", "first_downs", "rush_yds", "pass_yds",
        "total_yards", "turnovers", "pen_yds"
    ]

    for stat in diff_stats:
        df[f"diff_{stat}_avg"] = df[f"home_{stat}_avg"] - df[f"away_{stat}_avg"]

    df["home_team_wins"] = (df["home_score"] > df["away_score"]).astype(int)

    keep_cols = ["season", "week", "home_team", "away_team"] + [c for c in df.columns if c.startswith("diff_")] + ["home_team_wins"]
    df = df[keep_cols].dropna().reset_index(drop=True)

    return df


def get_test_train_split(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c.startswith("diff_")]
    X_set = df[feature_cols]
    y_set = df["home_team_wins"]

    return X_set, y_set
