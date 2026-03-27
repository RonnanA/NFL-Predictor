import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from config import PRODUCTION_DIR, TEST_DIR

FEATURE_COLS = (
    [f"diff_roll_{s}" for s in ["score", "rush_yds", "pass_yds", "total_yards", "first_downs", "turnovers", "pen_yds"]] +
    [f"diff_std_{s}"  for s in ["score", "rush_yds", "pass_yds", "total_yards", "first_downs", "turnovers", "pen_yds"]] +
    ["diff_streak", "home_rest_days", "away_rest_days"]
)


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  calibrated),
    ])


# ─────────────────────────────────────────────
# Cross validation
# ─────────────────────────────────────────────

def rolling_cross_validate(matchup_df: pd.DataFrame) -> list[dict]:
    seasons = sorted(matchup_df["season"].unique())

    # need at least 2 seasons to form one fold
    if len(seasons) < 2:
        raise ValueError(f"need at least 2 seasons, got {seasons}")

    folds = [
        (seasons[:i], seasons[i])
        for i in range(1, len(seasons))
    ]

    results = []
    for train_seasons, test_season in folds:
        train_df = matchup_df[matchup_df["season"].isin(train_seasons)]
        test_df  = matchup_df[matchup_df["season"] == test_season]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["home_team_wins"]
        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df["home_team_wins"]

        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        probs  = pipe.predict_proba(X_test)[:, 1]

        results.append({
            "train_seasons": train_seasons,
            "test_season":   test_season,
            "n_train":       len(X_train),
            "n_test":        len(X_test),
            "accuracy":      round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":       round(roc_auc_score(y_test, probs), 4),
            "brier_score":   round(brier_score_loss(y_test, probs), 4),
        })

        _print_fold_result(results[-1])

    _print_cv_summary(results)
    return results


def _print_fold_result(fold: dict) -> None:
    train_str = ", ".join(str(s) for s in fold["train_seasons"])
    print(f"\n  train [{train_str}] → test [{fold['test_season']}]")
    print(f"    n_train={fold['n_train']}  n_test={fold['n_test']}")
    print(f"    accuracy:    {fold['accuracy']:.2%}")
    print(f"    roc_auc:     {fold['roc_auc']:.3f}")
    print(f"    brier_score: {fold['brier_score']:.3f}")


def _print_cv_summary(results: list[dict]) -> None:
    accs   = [r["accuracy"]    for r in results]
    aucs   = [r["roc_auc"]     for r in results]
    briers = [r["brier_score"] for r in results]
    print(f"\n  CV summary ({len(results)} folds)")
    print(f"    accuracy:    {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    print(f"    roc_auc:     {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"    brier_score: {np.mean(briers):.3f} ± {np.std(briers):.3f}")


# ─────────────────────────────────────────────
# Train final model
# ─────────────────────────────────────────────

def train_final(matchup_df: pd.DataFrame, save_path=None) -> Pipeline:
    X = matchup_df[FEATURE_COLS]
    y = matchup_df["home_team_wins"]

    pipe = build_pipeline()
    pipe.fit(X, y)

    if save_path:
        joblib.dump(pipe, save_path)
        print(f"final model saved → {save_path}")

    return pipe


def load_model(path=None) -> Pipeline:
    path = path or PRODUCTION_DIR / "nfl_rf_model.pkl"
    return joblib.load(path)