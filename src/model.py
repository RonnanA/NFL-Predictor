import pandas as pd
from sklearn.ensemble import RandomForestClassifier #temp
from sklearn.metrics import accuracy_score


def get_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = df[df["season"] < 2024]
    test_df = df[df["season"] == 2024]

    feature_cols = [c for c in df.columns if c.startswith("diff_")]

    X_train = train_df[feature_cols]
    y_train = train_df["home_team_wins"]

    X_test = test_df[feature_cols]
    y_test = test_df["home_team_wins"]

    return X_train, y_train, X_test, y_test, test_df


def train__test_learn(X_train, y_train, X_test, y_test, test_df):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    #print(f"Model acc based on 2024 season test data: {acc:.2%}")

    probs = model.predict_proba(X_test)[:, 1]
    test_df.loc[:, "home_win_prob"] = probs
    print(test_df.head())
