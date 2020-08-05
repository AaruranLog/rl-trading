import pytest
from trading.system import *
from .test_system import Test_TradingEnv
from pandas_datareader._utils import RemoteDataError
from trading import filtered_tickers, blacklist

# def test_blacklist():
#     blacklist = open("../blacklist.txt", "r").read().split("\n")

#     for b in blacklist:
#         print(b)
#         with pytest.raises(Exception) as e_info:
#             Test_TradingEnv.basic_loop(b)


# def test_filtered():
#     with open("../filtered_tickers.txt", "r") as src:
#         filtered_tickers = src.read().split("\n")


def test_golden_blacklist():
    golden_blacklist = open("tests/golden_blacklist.txt", "r").read().split("\n")
    assert blacklist == golden_blacklist


def test_golden_filtered():
    golden_filtered_tickers = (
        open("tests/golden_filtered_tickers.txt", "r").read().split("\n")
    )
    assert golden_filtered_tickers == filtered_tickers
