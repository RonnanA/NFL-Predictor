from thefuzz import process
from teams import TEAM_NAME_TO_ALIAS, get_alias



def get_user_input(home_team: str, away_team: str) -> tuple[str, str]:

    team_name_list = list(TEAM_NAME_TO_ALIAS.keys())

    fuzzy_home_tuple = process.extractOne(home_team, team_name_list)
    fuzzy_away_tuple = process.extractOne(away_team, team_name_list)

    home_alias = get_alias(fuzzy_home_tuple[0])
    away_alias = get_alias(fuzzy_away_tuple[0])

    return home_alias, away_alias
