from src.scraper import scrape_season
from reformater import reformat_season
import pandas as pd


file_name = "Season-2020.csv"
df = scrape_season(2020)
reformat_season(df, file_name)

