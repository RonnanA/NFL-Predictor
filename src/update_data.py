import pandas as pd
from scraper import scrape_season, scrape_stats
from reformater import reformat_season
from merge_files import merge

def update_data_script(year: int):
    df = scrape_season(year)
    reformat_season(df, year)
    scrape_stats(year)
    merge(year)
    print("\033[32m----------------------\n" \
          " script complete twin\n" \
          "----------------------\033[0m")
    
update_data_script(2025)