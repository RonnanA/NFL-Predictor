import pandas as pd
import os

SEASON_2023 = "Season-2023.csv"
STATS_2023 = "Stats-2023.csv"
MERGED_FILE_NAME_2023 = "Merged-Season-2023.csv"

SEASON_2022 = "Season-2022.csv"
STATS_2022 = "Stats-2022.csv"
MERGED_FILE_NAME_2022 = "Merged-Season-2022.csv"

SEASON_2021 = "Season-2021.csv"
STATS_2021 = "Stats-2021.csv"
MERGED_FILE_NAME_2021 = "Merged-Season-2021.csv"

SEASON_2020 = "Season-2020.csv"
STATS_2020 = "Stats-2020.csv"
MERGED_FILE_NAME_2020 = "Merged-Season-2020.csv"


#def merge(season_year: int, stats_year: int):


# reverse dataframe
def reverse(season_df):
    reversed_season_df = season_df.iloc[::-1].reset_index(drop=True)
    return reversed_season_df

def merge_old(season_file, stats_file):

    # change dir and create dfs
    os.chdir(r"G:\Projects\NFL-Predictor\data\raw")
    season = pd.read_csv(season_file, header=0)
    stats = pd.read_csv(stats_file, header=0)

    # convert to datetime obj
    season["event_date"] = pd.to_datetime(season["event_date"])
    stats["event_date"] = pd.to_datetime(stats["event_date"])

    # drop cols
    # some cols are missing/different in the gathered data
    # "game_time"
    season = season.drop(columns=[
    "status","season","week","week_day",
    "tm_nano","tm_alt_market","tm_alt_alias",
    "opp_nano","opp_alt_market","opp_alt_alias","boxscore_stats_link"
    ])
    stats = stats.drop(columns=["season", "nano", "boxscore_stats_link"])

    # split stats into team1 and team2
    team1_stats = stats.merge(season[['event_date','tm_alias']], left_on=['event_date','alias'], right_on=['event_date','tm_alias'], how='inner')
    team2_stats = stats.merge(season[['event_date','opp_alias']], left_on=['event_date','alias'], right_on=['event_date','opp_alias'], how='inner')

    # get names of all stat cols
    stat_cols = [col for col in stats.columns if col not in ['event_date','market','name','alias']]

    # rename stat cols for each team
    team1_stats = team1_stats.rename(columns={c: f"tm_{c}" for c in stat_cols})
    team2_stats = team2_stats.rename(columns={c: f"opp_{c}" for c in stat_cols})

    # build team1 and team2 full dfs
    team1_stats = team1_stats[['event_date','tm_alias'] + [f"tm_{c}" for c in stat_cols]]
    team2_stats = team2_stats[['event_date','opp_alias'] + [f"opp_{c}" for c in stat_cols]]

    # merge team1 and team2 into final df
    final = season.merge(team1_stats, on=['event_date','tm_alias'], how='left')
    final = final.merge(team2_stats, on=['event_date','opp_alias'], how='left')

    # check for and add home_team_win flag to each game
    final['home_team_win'] = ((final['tm_score'] > final['opp_score']) & (final['tm_location']=='H') |
                              (final['tm_score'] < final['opp_score']) & (final['tm_location']=='A')).astype(int)

    # for incorrectly ordered seasons
    #final = reverse(final);

    # save to new merged csv file
    final.to_csv(rf"G:\Projects\NFL-Predictor\data\processed\{MERGED_FILE_NAME_2020}", index=False)
    print("Merged Successfully!")

    

#merge_old(SEASON_2020, STATS_2020)

