from playwright.sync_api import sync_playwright
from teams import get_pfr_code, get_alias_from_tc
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO
import pandas as pd
import time
import os
import signal
import sys
import random


def scrape_season(year: int) -> pd.DataFrame:
    output_path = rf"G:\Projects\NFL-Predictor\data\raw\Season-{year}.csv"
    if not os.path.exists(output_path):
        df = scrape_season_helper(year)
        return df
    
    df = pd.read_csv(output_path)
    try:
        first_empty_row = df[df.isnull().any(axis=1)].iloc[0]
        first_empty_row['event_date'] = pd.to_datetime(first_empty_row['event_date'], errors='coerce')

        latest_game = first_empty_row['event_date']
        today = datetime.today()

        if latest_game.date() <= today.date():
            updated_df = scrape_season_helper(year)
            return updated_df
            
    except:
        print(f"-- SEASON {year} FILE IS UP-TO-DATE --")    
        return None


def scrape_season_helper(year: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                page.goto(url, timeout=60000)
                html = page.content()
            except Exception as e:
                print(f"\033[31mERROR -- FAILED TO LOAD PAGE FOR {year}: {e} --\033[0m")
                return pd.DataFrame()
            
            finally:
                browser.close()

    except Exception as e:
        print(f"\033[31mERROR -- PLAYWRIGHT LAUNCH FAILED WITH EXCEPTION: {e} --\033[0m")
        return pd.DataFrame()

    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="games")
        if table is None:
            print(f"\033[31mERROR -- COULD NOT FIND TABLE FOR {year}: {e} --\033[0m")
            return pd.DataFrame()
        
        df = pd.read_html(StringIO(str(table)), header=None)[0]
    except Exception as e:
        print(f"\033[31mERROR -- FAILED TO PARSE HTML FOR {year}: {e} --\033[0m")
        return pd.DataFrame()
    
    print("\033[32m-----------------------------------\n" \
          " season web scrape successful twin\n" \
          "-----------------------------------\033[0m")
    return df


def scrape_stats(year: int):
    kill_now = False
    def handle_exit(sig, frame):
        nonlocal kill_now
        print("in handle exit func")
        kill_now = True

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    urls, alias_list = build_boxscore_urls(year)
    if len(urls) == 0:
        print("\033[31mERROR -- NO URLS GIVEN TO SCRAPE --\033[0m")
        return
    
    output_path = rf"G:\Projects\NFL-Predictor\data\raw\Stats-{year}.csv"
    min_delay = 3
    max_delay = 7

    if os.path.exists(output_path):
        final_df = pd.read_csv(output_path)
        already_done = set(final_df["url"].to_list())
        print(f"-- FOUND {len(already_done)} GAMES ALREADY SCRAPED --")
    else:
        final_df = pd.DataFrame()
        already_done = set()

    with Progress(SpinnerColumn(), 
                  TextColumn("[progress.description]{task.description}"), 
                  BarColumn(), 
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
                  transient=True
    ) as progress:
        num_scrapes = len(urls)
        task = progress.add_task("[green]Scraping...", total=num_scrapes)

        for url, alias_tuple in zip(urls, alias_list):
            if url in already_done:
                print(f"-- SKIPPING {url} (ALREADY SCRAPED) --")
                progress.update(task, advance=1)
                continue

            flat_df = scrape_stats_helper(url, alias_tuple[0], alias_tuple[1])
            if flat_df is not None:
                final_df = pd.concat([final_df, flat_df], ignore_index=True)
                final_df.to_csv(output_path, index=False)
                print(f"-- SCRAPED {url} --")
            else:
                print(f"\033[31m-- SKIPPED {url} (FAILED) --\033[0m")

            progress.update(task, advance=1)
            
            if kill_now:
                print("in kill not cond")
                sys.exit(0)

            delay = random.uniform(min_delay, max_delay)
            print(f"\033[33m-- SLEEPING {delay:.1f}s... --\033[0m")
            time.sleep(delay)

    print("\033[32m----------------------------------\n" \
          " stats web scrape successful twin\n" \
          "----------------------------------\033[0m")


def scrape_stats_helper(url: str, season_home_team: str, season_away_team: str) -> pd.DataFrame:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            html = page.content()
            browser.close()

            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", id="team_stats")
            if table is None:
                print(f"\033[31mERROR -- NO STATS TABLE SOUND AT {url} --\033[0m")
                return None
            
            df = pd.read_html(StringIO(str(table)), header=None)[0]
            flat_df = flatten_game_stats(df, season_home_team, season_away_team)
            flat_df["url"] = url
            return flat_df
        
    except Exception as e:
        print(f"\033[31mERROR -- FAILED TO SCRAPE {url}: {e} --\033[0m")
        return None


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


def build_boxscore_urls(year: int) -> tuple[list[str], list[tuple[str, str]]]:
    urls = []
    alias_list = []
    if (year < 1975) or (year > 2025):
        print("\033[31mERROR -- ENTER A VALID SEASON YEAR (1975-2025) --\033[0m")
        return urls, alias_list
    
    df = pd.read_csv(rf"G:\Projects\NFL-Predictor\data\raw\Season-{year}.csv")
    
    for row in df.itertuples(index=False):
        date_str = row.event_date.replace("-", "")
        #home_alias = row.tm_alias if row.tm_location == "H" else row.opp_alias
        home_code = get_pfr_code(row.home_team)
        alias_list.append((row.home_team, row.away_team))

        url = f"https://www.pro-football-reference.com/boxscores/{date_str}0{home_code}.htm"
        urls.append(url)

    return urls, alias_list
