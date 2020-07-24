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
        

def test_golden_blacklist():
    blacklist = open('blacklist.txt', 'r').read().split('\n')
    golden_blacklist = open('golden_blacklist.txt', 'r').read().split('\n')
    assert blacklist == golden_blacklist
    
def test_golden_filtered():
    filtered_tickers = open("filtered_tickers.txt", "r").read().split("\n")
    golden_filtered_tickers = open("golden_filtered_tickers.txt", "r").read().split("\n")
    assert golden_filtered_tickers == filtered_tickers