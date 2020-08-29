import pathlib

import pytest
from pandas.testing import assert_frame_equal
from pandas_datareader._utils import RemoteDataError

from trading import blacklist, filtered_tickers
from trading.system import *


def validate_numeric_list(x):
    assert not any(map(math.isnan, x)), "Found nan in x."
    return True


def basic_loop(t, *args, **kwargs):
    env = TradingEnv(ticker=t, *args, **kwargs)
    state = env.reset()
    validate_numeric_list(state)
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        assert len(state) == len(next_state)
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()
    assert_frame_equal(h, h.fillna(np.inf))


def test_blacklist():
    for b in blacklist:
        if len(b):
            print(b)
            with pytest.raises(Exception) as e_info:
                basic_loop(b)


def test_filtered():
    assert len(filtered_tickers)


def test_golden_blacklist():
    p = pathlib.Path(__file__).parent
    filename = p / "golden_blacklist.txt"
    golden_blacklist = filename.read_text().split("\n")
    assert blacklist == golden_blacklist


def test_golden_filtered():
    p = pathlib.Path(__file__).parent
    filename = p / "golden_filtered_tickers.txt"
    golden_filtered_tickers = filename.read_text().split("\n")
    assert golden_filtered_tickers == filtered_tickers
