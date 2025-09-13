import pandas as pd
import requests
from io import StringIO
import re

url = "https://www.pro-football-reference.com/boxscores/202409050kan.htm#team_stats"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

# Remove HTML comments so pandas can parse tables
clean_html = re.sub("<!--|-->", "", response.text)

tables = pd.read_html(StringIO(clean_html))
games = tables[0]
print(games.head())
