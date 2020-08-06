import pytest
from trading.system import *
from .test_system import Test_TradingEnv
from pandas_datareader._utils import RemoteDataError
from trading import filtered_tickers, blacklist
import pathlib

def test_blacklist():
    for b in blacklist:
        print(b)
        with pytest.raises(Exception) as e_info:
            Test_TradingEnv.basic_loop(b)


def test_filtered():
    assert len(filtered_tickers)


def test_golden_blacklist():
    p = pathlib.Path(__file__).parent
    filename = p / 'golden_blacklist.txt' 
    golden_blacklist = filename.read_text().split("\n")
    assert blacklist == golden_blacklist


def test_golden_filtered():
    p = pathlib.Path(__file__).parent
    filename = p / 'golden_filtered_tickers.txt' 
    golden_filtered_tickers = filename.read_text().split("\n")
    assert golden_filtered_tickers == filtered_tickers
