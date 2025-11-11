import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from config import TEST_DIR, PROCESSED_DIR

#TODO add mode to get different test sets
def test_train_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    train_df = df[df["season"] < 2024]
    test_df = df[df["season"] == 2024]

    feature_cols = [c for c in df.columns if c.startswith("diff_")]

    X_train = train_df[feature_cols]
    y_train = train_df["home_team_wins"]

    X_test = test_df[feature_cols]
    y_test = test_df["home_team_wins"]

    return X_train, y_train, X_test, y_test, test_df

#TODO improve model
def test_train_sklearn(X_train, y_train, X_test=None, y_test=None, test_df=None, mode="predict"):
    pipe = Pipeline([
       ('scaler', StandardScaler()) ,
       ('model', RandomForestClassifier(
           n_estimators=200,
           max_depth=None,
           random_state=42,
           n_jobs=1
       ))
    ])

    # Predict mode to train model on current data (DEFAULT)
    if(mode == "predict"):
        output_path = PROCESSED_DIR / "nfl_rf_model.pkl"

        pipe.fit(X_train, y_train)
        joblib.dump(pipe, output_path)
        print(f"model trained on full dataset and saved to {output_path} twin")

        return pipe
    

    # Evaluation mode to train and improve model
    elif(mode == "eval"):
        output_path = TEST_DIR / "test_nfl_rf_model.pkl"
        assert X_test is not None and y_test is not None, "X_test/y_test are required for eval mode"

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:,1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, probs)

        print(f"model accuracy: {acc:.2%}")
        print(f"ROC AUC: {auc:.3f}")

        test_df = test_df.copy() if test_df is not None else pd.DataFrame()
        test_df["home_win_prob"] = probs

        joblib.dump(pipe, output_path)
        print(f"test model saved to {output_path} twin")

        return pipe, test_df

    else:
        raise ValueError("Invalid mode. Use 'predict' or 'eval'.")