import pandas as pd
from teams import get_alias


def reformat_season(df: pd.DataFrame, year: int):
    if df is not None:  
        df = df.dropna(subset=["Week"])
        df = df[df["Week"] != "Week"]
        df = df[df["Date"] != "Playoffs"]

        df = df.rename(columns={'Week': 'week', 
                                'Day': 'week_day', 
                                'Date': 'event_date', 
                                'Time': 'game_time', 
                                'Winner/tie': 'winner', 
                                'Loser/tie': 'loser',
                                'PtsW': 'winner_score', 
                                'PtsL': 'loser_score',
                                'Unnamed: 5': 'at_symbol'
        })

        df["winner"] = df["winner"].apply(get_alias)
        df["loser"] = df["loser"].apply(get_alias)

        df["home_team"] = df.apply(
            lambda row: row["loser"] if row["at_symbol"] == "@" else row["winner"], axis=1
        )
        df["away_team"] = df.apply(
            lambda row: row["winner"] if row["at_symbol"] == "@" else row["loser"], axis=1
        )
        df["home_score"] = df.apply(
            lambda row: row["loser_score"] if row["at_symbol"] == "@" else row["winner_score"], axis=1
        )
        df["away_score"] = df.apply(
            lambda row: row["winner_score"] if row["at_symbol"] == "@" else row["loser_score"], axis=1
        )

        df = df[["week", "week_day", "event_date", "game_time",
                 "home_team", "away_team", "home_score", "away_score"]]

        df = df.reset_index(drop=True)

        df.to_csv(rf"G:\Projects\NFL-Predictor\data\raw\Season-{year}.csv", index=False)

        print("\033[32m---------------------------------------------------------------------------------------\n" \
            f" Data was reformatted twin. Path: 'G:\\Projects\\NFL-Predictor\\data\\raw\\Season-{year}.csv'\n" \
            "---------------------------------------------------------------------------------------\033[0m")
