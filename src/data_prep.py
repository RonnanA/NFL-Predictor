import pandas as pd


def load_data(train_files, test_file=None):
    df_train = pd.concat([pd.read_csv(f, parse_dates=["event_date"]) for f in train_files])
    df_test = pd.read_csv(test_file, parse_dates=["event_date"]) if test_file else None
    return df_train, df_test


def preprocess(df):
    postseason = ["WildCard", "Division", "ConfChamp", "SuperBowl"]

    dfcut = df.loc[~df["week"].isin(postseason)].copy()
    dfcut.loc[:, "home_team_wins"] = (dfcut["tm_location"] == "H").astype(int)
    dfcut = dfcut.sort_values(["tm_alias", "event_date"])

    rolling_stats = ["tm_score","opp_score","tm_total_yards","opp_total_yards",
                     "tm_turnovers","opp_turnovers"]
    
    for stat in rolling_stats:
        dfcut[f"{stat}_last3"] = (
            dfcut.groupby("tm_alias")[stat]
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
    return dfcut



df1, df2 = load_data(["data/processed/Merged-2020.csv", "data/processed/Merged-2021.csv"])
dfnew = preprocess(df1)
dfnew.to_csv("data/processed/test1.csv", index=False)