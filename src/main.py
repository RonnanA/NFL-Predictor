from scraper import scrape_season, scrape_stats, build_boxscore_urls
from reformater import reformat_season
import pandas as pd


season = 2020
df = scrape_season(season)
reformat_season(df, season)
#print(df.head())
