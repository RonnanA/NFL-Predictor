import pandas as pd
from teams import get_alias, get_alias_from_tc
from config import RAW_DIR


def reformat_season(df: pd.DataFrame, year: int):
    output_path = RAW_DIR / f"Season-{year}.csv"

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

        df.to_csv(output_path, index=False)

        print("\033[32m-------------------------------------------------------------------------------------\n" \
            f" Data was reformatted twin. Path: {output_path}\n" \
            "-------------------------------------------------------------------------------------\033[0m")

def flatten_game_stats(df: pd.DataFrame, season_home_team: str, season_away_team: str) -> pd.DataFrame:
    stats_tm_alias = get_alias_from_tc(df.columns[1])
    stats_opp_alias = get_alias_from_tc(df.columns[2])

    if stats_tm_alias == season_home_team and stats_opp_alias == season_away_team:
        df.columns = ['stat', season_home_team, season_away_team]
        home_col, away_col = season_home_team, season_away_team
    elif stats_tm_alias == season_away_team and stats_opp_alias == season_home_team:
        df.columns = ['stat', season_away_team, season_home_team]
        home_col, away_col = season_home_team, season_away_team
    else:
        print("\033[31mERROR -- SCRAPED ALIASES DO NOT MATCH SEASON FILE ALIASES --\033[0m")
        return
    
    flat_data = {}

    expanded_stats = {
        "Rush-Yds-TDs": ["rush_att", "rush_yds", "rush_tds"],
        "Cmp-Att-Yd-TD-INT": ["pass_cmp", "pass_att", "pass_yds", "pass_tds", "pass_int"],
        "Sacked-Yards": ["sacks", "sack_yds"],
        "Fumbles-Lost": ["fmbl", "fmbl_yds"],
        "Penalties-Yards": ["pen", "pen_yds"],
        "Third Down Conv.": ["third_d", "thrid_d_conv"],
        "Fourth Down Conv.": ["fourth_d", "fourth_d_conv"]
    }

    for _, row in df.iterrows():
        stat_name = row['stat']
        home_value = str(row[home_col])
        away_value = str(row[away_col])
        
        if stat_name in expanded_stats:
            keys = expanded_stats[stat_name]
            home_parts = home_value.split("-")
            away_parts = away_value.split("-")
            for k, home_part, away_part in zip(keys, home_parts, away_parts):
                flat_data[f"home_{k}"] = home_part
                flat_data[f"away_{k}"] = away_part
        else:
            flat_data[f"home_{stat_name.lower().replace(' ', '_')}"] = home_value
            flat_data[f"away_{stat_name.lower().replace(' ', '_')}"] = away_value

    flat_data["home_alias"] = season_home_team
    flat_data["away_alias"] = season_away_team

    return(pd.DataFrame([flat_data]))
