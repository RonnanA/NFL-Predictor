from scraper import scrape_season, scrape_stats, build_boxscore_urls
from reformater import reformat_season
import pandas as pd

season = 2020
urls, alias_list = build_boxscore_urls(season)
df = scrape_stats(urls, alias_list, season)

