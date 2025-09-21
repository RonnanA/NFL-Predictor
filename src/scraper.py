from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from teams import get_pfr_code, get_alias_from_tc
import pandas as pd
import time

def scrape_season(season: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{season}/games.htm"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                page.goto(url, timeout=60000)
                html = page.content()
            except Exception as e:
                print(f"ERROR: Failed to load page for {season} with exception {e}")
                return pd.DataFrame()
            
            finally:
                browser.close()

    except Exception as e:
        print(f"ERROR: Playwright launch failed with exception {e}")
        return pd.DataFrame()

    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="games")
        if table is None:
            print(f"ERROR: Could not find table for {season} with exception {e}")
            return pd.DataFrame()
        
        df = pd.read_html(str(table), header=None)[0]
    except Exception as e:
        print(f"ERROR: Failed to parse HTML for {season} with exception {e}")
        return pd.DataFrame()
    
    print("----------------------------\n" \
          " Web scrape successful twin\n" \
          "----------------------------")
    return df


def scrape_stats(url):

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        html = page.content()
        browser.close()

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="team_stats")
        df = pd.read_html(str(table), header=None)[0]
        
        return df


def flatten_game_stats(df: pd.DataFrame, season_tm_alias, season_opp_alias):
    
    stats_tm_alias = get_alias_from_tc(df.columns[1])
    stats_opp_alias = get_alias_from_tc(df.columns[2])

    if stats_tm_alias == season_tm_alias:
        tm_col = stats_tm_alias
        opp_col = stats_opp_alias
        df.columns = ['stat', tm_col, opp_col]
    else:
        tm_col = stats_opp_alias
        opp_col = stats_tm_alias
        df.columns = ['stat', opp_col, tm_col]
    
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
        tm_value = str(row[tm_col])
        opp_value = str(row[opp_col])
        
        if stat_name in expanded_stats:
            keys = expanded_stats[stat_name]
            tm_parts = tm_value.split("-")
            opp_parts = opp_value.split("-")
            for k, tm_part, opp_part in zip(keys, tm_parts, opp_parts):
                flat_data[f"tm_{k}"] = tm_part
                flat_data[f"opp_{k}"] = opp_part
        else:
            flat_data[f"tm_{stat_name.lower().replace(' ', '_')}"] = tm_value
            flat_data[f"opp_{stat_name.lower().replace(' ', '_')}"] = opp_value

    return(pd.DataFrame([flat_data]))


def build_boxscore_urls(season: int):
    df = pd.read_csv(rf"G:\Projects\NFL-Predictor\data\raw\Season-{season}.csv")

    urls = []
    for row in df.itertuples(index=False):
        date_str = row.event_date.replace("-", "")
        home_alias = row.tm_alias if row.tm_location == "H" else row.opp_alias
        home_code = get_pfr_code(home_alias)

        url = f"https://www.pro-football-reference.com/boxscores/{date_str}0{home_code}.htm"
        urls.append(url)

    return urls


df = scrape_stats("https://www.pro-football-reference.com/boxscores/202009130jax.htm")
flat = flatten_game_stats(df, "JAC", "IND")
#flat.to_csv(rf"G:\Projects\NFL-Predictor\data\raw\testres.csv", index=False)