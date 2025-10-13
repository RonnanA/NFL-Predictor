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