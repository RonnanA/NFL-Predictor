import pandas as pd
from sklearn.pipeline import Pipeline
from model import FEATURE_COLS
from features import STAT_COLS


# ─────────────────────────────────────────────
# Single matchup
# ─────────────────────────────────────────────

def make_prediction(
    home_team: str,
    away_team: str,
    pipe: Pipeline,
    snapshots_df: pd.DataFrame,
) -> dict | None:
    home_snap = get_latest_snapshot(snapshots_df, home_team)
    away_snap = get_latest_snapshot(snapshots_df, away_team)

    if home_snap is None:
        return {"error": f"no snapshot found for {home_team}"}
    if away_snap is None:
        return {"error": f"no snapshot found for {away_team}"}

    X = _build_matchup_row(home_snap, away_snap)
    prob = float(pipe.predict_proba(X)[:, 1][0])

    winner     = home_team if prob >= 0.5 else away_team
    confidence = prob if prob >= 0.5 else 1 - prob

    return {
        "home_team":        home_team,
        "away_team":        away_team,
        "home_win_prob":    round(prob, 4),
        "predicted_winner": winner,
        "confidence":       round(confidence, 4),
    }


# ─────────────────────────────────────────────
# Full week of games (for the GUI)
# ─────────────────────────────────────────────

def predict_week(
    matchups: list[dict],
    pipe: Pipeline,
    snapshots_df: pd.DataFrame,
) -> list[dict]:
    """
    Predict a full slate of games. Each matchup is a dict with
    keys 'home_team' and 'away_team'.

    Example:
        matchups = [
            {"home_team": "KC",  "away_team": "BUF"},
            {"home_team": "PHI", "away_team": "DAL"},
        ]
    """
    results = []
    for m in matchups:
        result = make_prediction(m["home_team"], m["away_team"], pipe, snapshots_df)
        results.append(result)
    return results


# ─────────────────────────────────────────────
# Snapshot lookup
# ─────────────────────────────────────────────

def get_latest_snapshot(snapshots_df: pd.DataFrame, team: str) -> pd.Series | None:
    team_rows = snapshots_df[snapshots_df["team"] == team]
    if team_rows.empty:
        return None
    return team_rows.sort_values(["season", "week"]).iloc[-1]


# ─────────────────────────────────────────────
# Feature row assembly
# ─────────────────────────────────────────────

def _build_matchup_row(home_snap: pd.Series, away_snap: pd.Series) -> pd.DataFrame:
    """
    Build the feature row the model expects from two team snapshots.
    Mirrors the logic in build_matchup_frame() in features.py so
    training and prediction are always consistent.
    """
    row = {}

    for stat in STAT_COLS:
        row[f"diff_roll_{stat}"] = home_snap[f"roll_{stat}"] - away_snap[f"roll_{stat}"]
        row[f"diff_std_{stat}"]  = home_snap[f"std_{stat}"]  - away_snap[f"std_{stat}"]

    row["diff_streak"]     = home_snap["streak"] - away_snap["streak"]
    row["home_rest_days"]  = home_snap["rest_days"]
    row["away_rest_days"]  = away_snap["rest_days"]

    # enforce column order — must match exactly what the model was trained on
    return pd.DataFrame([row])[FEATURE_COLS]