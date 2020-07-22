import pytest
from system import *
from pandas_datareader._utils import RemoteDataError
import numpy as np
import math


def validate_state(state):
    assert not (np.nan in state), "Found nan in state."

def test_create_env():
    env = TradingEnv(mode="dev")
    env.reset()
    _ = env.step(1)
    assert len(env.prices) == len(env.data["x"])
    _ = env.data.columns


def basic_loop(t):
    env = TradingEnv(ticker=t)
    state = env.reset()
    validate_state(state)
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        validate_state(next_state)
        assert len(state) == len(next_state)
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()
    

def test_apple_download():
    basic_loop("AAPL")

def test_abbv_download():
    basic_loop("ABBV")

    
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
    validate_state(state)

        
def test_all_states_valid():
    env = TradingEnv()
    state = env.reset()
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        validate_state(state)
        
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()

def test_all_states_valid_dev():
    env = TradingEnv(mode="dev")
    state = env.reset()
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        validate_state(state)
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()
    
    
def test_state_valid_text():
    env = TradingWithRedditEnv(mode="dev")
    state, text = env.reset()
    validate_state(state)

def basic_loop_with_text(t):
    env = TradingWithRedditEnv(ticker=t)
    state, text_vectors = env.reset()
    assert isinstance(text_vectors, list), f"Expected list, but got {type(vectors)}"
    done = False
    while not done:
        action = np.random.randint(low=-1, high=2)
        (next_state, next_text), r, done, _ = env.step(action)
        assert len(state) == len(next_state)
        assert isinstance(next_text, list), f"Expected list, but got {type(vectors)}"
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()


def test_reddit_loop():
    basic_loop_with_text("aapl")
    basic_loop_with_text("tsla")
