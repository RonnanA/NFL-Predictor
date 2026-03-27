import pandas as pd
from config import PROCESSED_DIR, RAW_DIR


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
    

def get_master_df(year: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"nfl-{year}.csv"
    if not path.exists():
        print(f"warning: no data file found for {year}, skipping")
        return pd.DataFrame()
    return pd.read_csv(path)
