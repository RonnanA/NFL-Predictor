import time
from rich.progress import Progress

with Progress(transient=True) as progress:
    task = progress.add_task("[green]Downloading...", total=50)
    for i in range(50):
        print("downloaded", i)
        time.sleep(0.2)
        progress.update(task, advance=1)

