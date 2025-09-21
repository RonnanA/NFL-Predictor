import pandas as pd
from teams import get_alias

PASTE_SEASON_2024 = "CSV-Paste-Season-2024.csv"
FORMATTED_SEASON_2024 = "Season-2024.csv"

PASTE_STATS_2024 = "CSV-Paste-Stats-2024.csv"
FORMATTED_STATS_2024 = "Stats-2024.csv"


def reformat_season(df: pd.DataFrame, season: int):
    if df is not None:  
        df = df.dropna(subset=["Week"])
        df = df[df["Week"] != "Week"]
        df = df[df["Date"] != "Playoffs"]

        df = df.rename(columns={'Week': 'week', 'Day': 'week_day', 'Date': 'event_date', 'Time': 'game_time', 'Winner/tie': 'tm_alias', 'Loser/tie': 'opp_alias',
                                'PtsW': 'tm_score', 'PtsL': 'opp_score'})

        df["tm_alias"] = df["tm_alias"].apply(get_alias)
        df["opp_alias"] = df["opp_alias"].apply(get_alias)

        df["tm_location"] = df.iloc[:, 5].apply(lambda x: "A" if x == "@" else "H")
        df["opp_location"] = df["tm_location"].apply(lambda x: "H" if x == "A" else "A")


        df = df.drop(df.columns[5], axis=1)
        df = df.drop(df.columns[6], axis=1)
        df = df.drop(df.columns[8:12], axis=1)
        df = df.reset_index(drop=True)

        print("---------------------------------------------------------------------------------------\n" \
            f" Data was reformatted twin. Path: 'G:\\Projects\\NFL-Predictor\\data\\raw\\Season-{season}.csv'\n" \
            "---------------------------------------------------------------------------------------")
        df.to_csv(rf"G:\Projects\NFL-Predictor\data\raw\Season-{season}.csv", index=False)

