import pandas as pd
from sklearn.pipeline import Pipeline

#TODO
def make_prediction(home_team: str, away_team: str, pipe: Pipeline, master_df: pd.DataFrame):
    home_latest = get_latest_team_row(master_df, home_team)
    away_latest = get_latest_team_row(master_df, away_team)

    if home_latest.empty or away_latest.empty:
        print(f"missing data for {home_team} or {away_team} twin")
        return None
    
    diff_features = {}
    for col in master_df.columns:
        if col.startswith("diff_"):
            home_val = home_latest[col].values[0]
            away_val = away_latest[col].values[0]
            diff_features[col] = home_val - away_val

    X_matchup = pd.DataFrame([diff_features])

    prob = pipe.predict_proba(X_matchup)[:, 1][0]

    print(f"{away_team} @ {home_team} -> predidcted home win probability: {prob:.1%}")


def get_latest_team_row(master_df: pd.DataFrame, team: str):
    return (
        master_df[
            (master_df["home_team"] == team) | (master_df["away_team"] == team)
        ].sort_values(["season", "week"]).tail(1)
    )