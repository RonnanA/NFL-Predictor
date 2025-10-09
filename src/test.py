import time
from rich.progress import Progress
from scraper import scrape_stats_helper
import pandas as pd
import sys
import signal

def test_prog_bar():
    with Progress(transient=True) as progress:
        task = progress.add_task("[green]Downloading...", total=50)
        for i in range(50):
            print("downloaded", i)
            time.sleep(0.2)
            progress.update(task, advance=1)

stop = False

def handle_exit(sig, frame):
    global stop
    print("in handler func")
    stop = True

signal.signal(signal.SIGINT, handle_exit)

def test_scrape():
    global stop
    url = "https://www.pro-football-reference.com/boxscores/202009100kan.htm"

    for i in range(25):
        if stop:
            print("stop req, exiting")
            sys.exit(0)

        print("start scrape", i)
        df = scrape_stats_helper(url, "KC", "HOU")
        print("scrape done")

        if stop:
            print("stop req, exiting")
            sys.exit(0)
            
        print("sleeping")
        time.sleep(2)

test_scrape()