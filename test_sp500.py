import pytest
from system import *
from test_system import basic_loop
from pandas_datareader._utils import RemoteDataError


def test_blacklist():
    blacklist = open('blacklist.txt', 'r').read().split('\n')
    
    for b in blacklist:
        print(b)
        with pytest.raises(Exception) as e_info:
            basic_loop(b)
            print(e_info)
            
def test_filtered():
    with open("filtered_tickers.txt", "r") as src:
        filtered_tickers = src.read().split("\n")