from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd

def scrape_season(season: int) -> pd.DataFrame:
    url = f"https://www.pro-football-reference.com/years/{season}/games.htm"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="games")
    df = pd.read_html(str(table), header=None)[0]
    
    print("----------------------------\n" \
          " Web scrape successful twin\n" \
          "----------------------------")
    return df

#def scrape_stats(season: int) -> pd.DataFrame:
    url
