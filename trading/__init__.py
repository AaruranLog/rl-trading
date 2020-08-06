import pathlib

parent_dir = pathlib.Path(__file__).parent

filtered_tickers_file = parent_dir / "filtered_tickers.txt"
blacklist_file = parent_dir / "blacklist.txt"

filtered_tickers = filtered_tickers_file.open().read().split("\n")
blacklist = blacklist_file.open().read().split("\n")

# filtered_tickers = open("/home/aaruran/Documents/Git/rl-trading/trading/filtered_tickers.txt", "r").read().split("\n")
# blacklist = open("/home/aaruran/Documents/Git/rl-trading/trading/blacklist.txt", "r").read().split("\n")
