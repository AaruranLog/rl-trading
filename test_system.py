import pytest
from system import TradingEnv
from pandas_datareader._utils import RemoteDataError
import numpy as np
def test_create_env():
    env = TradingEnv()
    env.reset()
    _ = env.step(1)
    assert len(env.prices) == len(env.data['mean'])
    _ = env.data.columns

def basic_loop(t):
    env = TradingEnv(ticker=t)
    state = env.reset()
    done = False
    np.random.seed(885)
    while not done:
    # for i in range(14):
        action = np.random.randint(low=-1, high=2)
        next_state, r, done, _ = env.step(action)
        assert len(state) == len(next_state)

# basic_loop_test('AAPL')
def test_stock_name_file_exists():
    ticker_list = []
    with open('./small_stock_name.txt') as src:
        ticker_list = src.read().split()
    assert len(ticker_list) == 82

def test_apple_download():
    basic_loop('AAPL')
    
def test_CELG_download_fails():
    with pytest.raises(RemoteDataError):
        basic_loop('CELG')
        
def test_action_error_float():
    env = TradingEnv()
    env.reset()
    with pytest.raises(AssertionError):
        env.step(0.99999)
        
def test_action_error_list():
    env = TradingEnv()
    env.reset()
    with pytest.raises(AssertionError):
        env.step([0])
        
def test_action_error_list():
    env = TradingEnv()
    env.reset()
    with pytest.raises(AssertionError):
        env.step([0])        
        
def test_action_succeeds():
    env = TradingEnv()
    env.reset()
    _ = env.step(1)
    _ = env.step(-1)
    _ = env.step(0)
    _ = env.step(0)
    _ = env.step(-1)
    _ = env.step(-1)

    
# def test_filtered_stocks():
#     ticker_list = []
#     with open('./small_stock_name.txt') as src:
#         ticker_list = src.read().split()
#     assert len(ticker_list) == 82
#     filtered_tickers = []
#     for i, t in enumerate(ticker_list):
#         try:
#             basic_loop_test(t)
#             filtered_tickers.append(t)
#         except:
# #             print(f'{t} failed')
#             continue
# #     print(f'\nTickers preserved: {len(filtered_tickers)} / {len(ticker_list)}')
#     assert len(filtered_tickers) > 0, f'Full ticker length = {len(ticker_list)}'
#     assert len(filtered_tickers) == 73
