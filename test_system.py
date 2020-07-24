import pytest
from system import *
from pandas_datareader._utils import RemoteDataError
import numpy as np
import math


filtered_tickers = open("filtered_tickers.txt", "r").read().split("\n")

def test_tickers():
    assert len(filtered_tickers)

def validate_numeric_list(x):
    assert not any(map(math.isnan, x)), "Found nan in x."
    return True

    
def test_continuous_actions():
    actions = np.linspace(-1, 1, 100)
    np.random.seed(885)
    np.random.shuffle(actions)
#     e = ContinuousTradingEnv('AAPL', 10, "train")
    e = ContinuousTradingEnv()
    for a in actions:
        e.reset()
        _ = e.step(a)
        
def basic_loop_continuous(t):
    env = ContinuousTradingEnv(ticker=t)
    state = env.reset()
    validate_numeric_list(state)
    done = False
    np.random.seed(885)
    while not done:
        action = np.random.uniform(low=-1.0, high=1.0)
        next_state, r, done, _ = env.step(action)
        assert len(state) == len(next_state)
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()
    assert all(h == h.dropna()), "Invalid history"
    
def test_cnts_loop():
    basic_loop_continuous('AAPL')
    basic_loop_continuous('TSLA')

def test_create_env():
    env = TradingEnv(mode="train")
    env.reset()
    _ = env.step(1)
    assert len(env.prices) == len(env.data["x"])
    _ = env.data.columns

def basic_loop(t):
    env = TradingEnv(ticker=t)
    state = env.reset()
    validate_numeric_list(state)
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        assert len(state) == len(next_state)
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()
    assert all(h == h.dropna()), "Invalid history"

@pytest.mark.incremental
class TestValidData:
    def validate_all_data(self, t):
        e = TradingEnv(ticker=t)
        start_index = e.df_initial_index - e.WINDOW_SIZE
        final_index = e.df_final_index 
        trading_data = e.data[start_index : final_index + 1]
        data_has_no_NaNs = all(trading_data == trading_data.dropna())
        assert data_has_no_NaNs, "NaNs found"
        
        assert trading_data['std'].min() > 0, f"Zero standard deviation found in {t} data."
    
    def test_mmm_download(self):
        self.validate_all_data('MMM')
        basic_loop("MMM")
    
    def test_apple_download(self):
        self.validate_all_data('AAPL')
        basic_loop("AAPL")

    def test_abbv_download(self):
        self.validate_all_data('ABBV')
        basic_loop("ABBV")

    def test_amzn_download(self):
        self.validate_all_data('AMZN')
        basic_loop("AMZN")
        
    def test_br_download(self):
        if 'BR' in filtered_tickers:
            self.validate_all_data('BR')
            basic_loop("BR")
    
    def test_all_tickers_download_and_valid(self):
        errors = {}
        for t in filtered_tickers:
            try:
                e = TradingEnv(ticker=t)
                start_index = e.df_initial_index - e.WINDOW_SIZE
                final_index = e.df_final_index 
                trading_data = e.data[start_index : final_index + 1]
                data_has_no_NaNs = all(trading_data == trading_data.dropna())
                assert data_has_no_NaNs, "NaNs found"
                assert trading_data['std'].min() > 0, f"Zero standard deviation found in {t} data."
            except AssertionError as ex:
                    errors[t] = ex
        if len(errors):
            raise AssertionError(f'Failed {len(errors)}/{len(filtered_tickers)}, {errors.keys()}')
    

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
    validate_numeric_list(state)

        
def test_all_states_valid():
    env = TradingEnv()
    state = env.reset()
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        validate_numeric_list(state)
        
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()

def test_all_states_valid_dev():
    env = TradingEnv(mode="dev")
    state = env.reset()
    done = False
    while not done:
        action = 0
        next_state, r, done, _ = env.step(action)
        validate_numeric_list(state)
    assert len(env.returns_list) == len(env.actions_list)
    h = env.close()
    
    
def test_state_valid_text():
    env = TradingWithRedditEnv(mode="dev")
    state, text = env.reset()
    validate_numeric_list(state)

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
    assert all(h == h.dropna()), "Invalid history"

def test_reddit_loop():
    basic_loop_with_text("aapl")
    basic_loop_with_text("tsla")

