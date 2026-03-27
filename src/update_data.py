import joblib
from config import YEAR, PRODUCTION_DIR, TEST_DIR
from scraper import scrape_season, scrape_stats
from reformater import reformat_season
from merge_files import merge, get_master_df
from model import test_train_split, test_train_sklearn


def refresh_data():
    df = scrape_season(YEAR)
    reformat_season(df, YEAR)
    scrape_stats(YEAR)
    merge(YEAR)

def load_master(years=None):
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]
    return get_master_df(years)

def get_or_train_model(master_df, *, model_path, evaluate=False):
    X_train, y_train, X_test, y_test, test_df = test_train_split(master_df)

    if model_path.exists():
        pipe = joblib.load(model_path)
    else:
        pipe = test_train_sklearn(X_train, y_train, X_test, y_test)
        joblib.dump(pipe, model_path)

    if evaluate:
        return pipe, test_df

    return pipe

# Prediction script
def setup_for_predict(update_data=False, update_model=False):
    if update_data:
        refresh_data()

    model_path = PRODUCTION_DIR / "nfl_rf_model.pkl"
    if update_model:
        model_path.unlink(missing_ok=True)  # force retrain

    master_df = load_master()
    pipe = get_or_train_model(master_df, model_path=model_path)
    return pipe, master_df

# Evaluation script
def setup_for_eval(update_data=False, update_model=False):
    if update_data:
        refresh_data()

    model_path = TEST_DIR / "test_nfl_rf_model.pkl"
    if update_model:
        model_path.unlink(missing_ok=True)

    master_df = load_master()
    pipe, test_df = get_or_train_model(master_df, model_path=model_path, evaluate=True)
    return pipe, master_df, test_df
