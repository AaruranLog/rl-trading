import pathlib

parent_dir = pathlib.Path(__file__).parent

filtered_tickers_file = parent_dir / "filtered_tickers.txt"
blacklist_file = parent_dir / "blacklist.txt"

filtered_tickers = filtered_tickers_file.open().read().split("\n")
blacklist = blacklist_file.open().read().split("\n")