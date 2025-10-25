import pandas as pd
from config import PROCESSED_DIR, RAW_DIR
from features import data_cleanup, build_features


def merge(year: int):
    input_path_season = RAW_DIR / f"Season-{year}.csv"
    input_path_stats = RAW_DIR / f"Stats-{year}.csv"
    output_path = PROCESSED_DIR / f"nfl-{year}.csv"
    
    season_df = pd.read_csv(input_path_season)
    stats_df = pd.read_csv(input_path_stats)

    stats_df = stats_df.drop(columns=["home_alias", "away_alias"])

    final_df = pd.concat([season_df, stats_df], axis=1)

    final_df.to_csv(output_path, index=False)
    print("\033[32m-----------------------\n" \
          " merge successful twin\n" \
          "-----------------------\033[0m")
    

def get_master_df(based_on=3) -> pd.DataFrame:
    years = [2020, 2021, 2022, 2023, 2024]

    df_list = []
    for year in years:
        tmp = pd.read_csv(PROCESSED_DIR / f"nfl-{year}.csv")
        clean = data_cleanup(tmp, year)
        df_list.append(clean)
    
    df = pd.concat(df_list, ignore_index=True)
    final = build_features(df, based_on)

    return final
