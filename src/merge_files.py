import pandas as pd


def merge(year: int):
    season_df = pd.read_csv(rf"G:\Projects\NFL-Predictor\data\raw\Season-{year}.csv")
    stats_df = pd.read_csv(rf"G:\Projects\NFL-Predictor\data\raw\Stats-{year}.csv")

    stats_df = stats_df.drop(columns=["home_alias", "away_alias"])

    final_df = pd.concat([season_df, stats_df], axis=1)

    final_df.to_csv(rf"G:\Projects\NFL-Predictor\data\processed\Merged-{year}.csv", index=False)
    print("\033[32m-----------------------\n" \
          " merge successful twin\n" \
          "-----------------------\033[0m")
