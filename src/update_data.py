import joblib
from config import YEAR, PRODUCTION_DIR, TEST_DIR
from scraper import scrape_season, scrape_stats
from reformater import reformat_season
from merge_files import merge, get_master_df
from model import test_train_split, test_train_sklearn


def setup_script(mode="predict", update_data=False, update_model=False):
    if update_data:
        df = scrape_season(YEAR)
        reformat_season(df, YEAR)
        scrape_stats(YEAR)
        merge(YEAR)
    
    master_df = get_master_df()

    X_train, y_train, X_test, y_test, test_df = test_train_split(master_df)

    if mode == "predict":
        if not update_model:
            pipe = joblib.load(PRODUCTION_DIR / "nfl_rf_model.pkl")
        else:
            pipe = test_train_sklearn(X_train, y_train)
    
    elif mode == "eval":
        if not update_model:
            pipe = joblib.load(TEST_DIR / "test_nfl_rf_model.pkl")
        else:
            pipe, test_df = test_train_sklearn(X_train, y_train, X_test, y_test, test_df, mode="eval")
            return pipe, master_df, test_df

    else:
        print(f"{mode} not 'eval' or 'predict'")
        return

    return pipe, master_df