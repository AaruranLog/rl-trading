import pytest
from system import *
from test_system import basic_loop
from pandas_datareader._utils import RemoteDataError


def test_blacklist():
    blacklist = open('blacklist.txt', 'r').read().split()
    
    for b in blacklist:
        print(b)
        with pytest.raises(Exception) as e_info:
            basic_loop(b)
            print(e_info)