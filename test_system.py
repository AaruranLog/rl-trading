import pytest
from system import *
from pandas_datareader._utils import RemoteDataError
import numpy as np
import math
from unittest import TestCase


def test_create_env():
    env = TradingEnv(mode="dev")
    env.reset()
    _ = env.step(1)
    assert len(env.prices) == len(env.data["mean"])
    _ = env.data.columns


def basic_loop(t):
    env = TradingEnv(ticker=t)
    state = env.reset()
    done = False
    np.random.seed(885)
    while not done:
        action = np.random.randint(low=-1, high=2)
        next_state, r, done, _ = env.step(action)
        assert len(state) == len(next_state)


def test_apple_download():
    basic_loop("AAPL")


def test_CELG_download_fails():
    with pytest.raises(RemoteDataError):
        basic_loop("CELG")


def test_action_error_float():
    env = TradingEnv(mode="dev")
    env.reset()
    with pytest.raises(AssertionError):
        env.step(0.99999999999)


def test_action_error_list():
    env = TradingEnv(mode="dev")
    env.reset()
    with pytest.raises(AssertionError):
        env.step([0])


def test_action_error_list():
    env = TradingEnv(mode="dev")
    env.reset()
    with pytest.raises(AssertionError):
        env.step([0])


def test_action_succeeds():
    env = TradingEnv(mode="dev")
    env.reset()
    _ = env.step(1)
    _ = env.step(-1)
    _ = env.step(0)
    _ = env.step(0)
    _ = env.step(-1)
    _ = env.step(-1)


def test_reward():
    env = TradingEnv(mode="dev")
    env.reset()
    _, R, __, ___ = env.step(1)
    assert isinstance(R, float)


def test_state_valid():
    env = TradingEnv(mode="dev")
    state = env.reset()
    for v in state:
        assert not math.isnan(v), "Found nan in state."


def test_state_valid_text():
    env = TradingWithRedditEnv(mode="dev")
    state, text = env.reset()
    for v in state:
        assert not math.isnan(v), "Found nan in state."
        

def basic_loop_with_text(t):
    env = TradingWithRedditEnv(ticker=t)
    state, text_vectors = env.reset()
    assert isinstance(text_vectors, list), f"Expected list, but got {type(vectors)}"
    done = False
    np.random.seed(885)
    while not done:
        action = np.random.randint(low=-1, high=2)
        (next_state, next_text), r, done, _ = env.step(action)
        assert len(state) == len(next_state)
        assert isinstance(next_text, list), f"Expected list, but got {type(vectors)}"
        
def test_reddit_loop():
    basic_loop_with_text('aapl')
    basic_loop_with_text('tsla')