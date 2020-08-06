import pytest
from trading.system import *
from pandas_datareader._utils import RemoteDataError
from pandas.testing import assert_frame_equal
import numpy as np
import math
from trading import filtered_tickers


def test_tickers():
    assert len(filtered_tickers)


def validate_numeric_list(x):
    assert not any(map(math.isnan, x)), "Found nan in x."
    return True

@pytest.mark.incremental
class Test_TradingEnv:
    Constructor = TradingEnv

    def test_create_env(self):
        env = self.Constructor(mode="train")
        env.reset()
        env.seed(seed=885)
        _ = env.step(1)
        assert len(env.prices) == len(env.data["x"])
        _ = env.data.columns

    def test_state_valid(self):
        env = self.Constructor(mode="dev")
        state = env.reset()
        validate_numeric_list(state)

    def basic_loop(self, t, *args, **kwargs):
        env = self.Constructor(ticker=t, *args, **kwargs)
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

    #         assert all(h == h.fillna(np.inf)), "Invalid history"

    def test_CELG_download_fails(self):
        with pytest.raises(RemoteDataError):
            self.basic_loop(t="CELG")

    def test_action_error_float(self):
        env = self.Constructor(mode="dev")
        env.reset()
        with pytest.raises(AssertionError):
            env.step(0.99999999999)

    def test_action_error_list(self):
        env = self.Constructor(mode="dev")
        env.reset()
        with pytest.raises(AssertionError):
            env.step([0])

    def test_action_error_list(self):
        env = self.Constructor(mode="dev")
        env.reset()
        with pytest.raises(AssertionError):
            env.step([0])

    def test_action_succeeds(self):
        env = self.Constructor(mode="dev")
        env.reset()
        _ = env.step(1)
        _ = env.step(-1)
        _ = env.step(0)
        _ = env.step(0)
        _ = env.step(-1)
        _ = env.step(-1)

    def test_reward(self):
        env = self.Constructor(mode="dev")
        env.reset()
        _, R, __, ___ = env.step(1)
        assert isinstance(R, float)
        
    def test_prices(self):
        env = self.Constructor(mode='dev')
        for i in range(-env.WINDOW_SIZE+1, env.WINDOW_SIZE-1):
            assert not math.isnan(env._get_raw_price(diff=i))
            assert not math.isnan(env._get_normalized_price(diff=i))
        
    def test_invalid_mode(self):
        with pytest.raises(AssertionError):
            env = self.Constructor(mode='?')
        env = self.Constructor()
        with pytest.raises(ValueError):
            env.get_time_endpoints('?')


@pytest.mark.incremental
class Test_CntsEnv(Test_TradingEnv):
    Constructor = ContinuousTradingEnv

    def test_action_error_float(self):
        actions = np.linspace(-1, 1, 100)
        np.random.seed(885)
        np.random.shuffle(actions)
        #     e = ContinuousTradingEnv('AAPL', 10, "train")
        e = ContinuousTradingEnv()
        for a in actions:
            e.reset()
            _ = e.step(a)

    def test_action_error_list(self):
        pass

    def basic_loop(self, t):
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
        assert_frame_equal(h, h.fillna(np.inf))

    #         assert all(h == h.fillna(np.inf)), "Invalid history"

    def test_cnts_loop(self):
        self.basic_loop("AAPL")
        self.basic_loop("TSLA")


class Test_RedditEnv(Test_TradingEnv):
    Constructor = TradingWithRedditEnv

    def test_state_valid(self):
        env = TradingWithRedditEnv(mode="dev")
        state, text = env.reset()
        validate_numeric_list(state)

    def basic_loop(self, t, *args, **kwargs):
        env = TradingWithRedditEnv(ticker=t)
        state, text_vectors = env.reset()
        assert isinstance(text_vectors, list), f"Expected list, but got {type(vectors)}"
        done = False
        while not done:
            action = np.random.randint(low=-1, high=2)
            (next_state, next_text), r, done, _ = env.step(action)
            assert len(state) == len(next_state)
            assert isinstance(
                next_text, list
            ), f"Expected list, but got {type(vectors)}"
        assert len(env.returns_list) == len(env.actions_list)
        h = env.close()
        assert_frame_equal(h, h.fillna(np.inf))

    def test_reddit_loop(self):
        self.basic_loop("aapl")
        self.basic_loop("tsla")


@pytest.mark.incremental
class TestValidData:
    def validate_all_data(self, t):
        e = TradingEnv(ticker=t)
        start_index = e.df_initial_index - e.WINDOW_SIZE
        final_index = e.df_final_index
        trading_data = e.data[start_index : final_index + 1]
        assert_frame_equal(trading_data, trading_data.fillna(np.inf))

        assert (
            trading_data["std"].min() > 0
        ), f"Zero standard deviation found in {t} data."

    def test_mmm_download(self):
        self.validate_all_data("MMM")

    def test_apple_download(self):
        self.validate_all_data("AAPL")

    def test_abbv_download(self):
        self.validate_all_data("ABBV")

    def test_amzn_download(self):
        self.validate_all_data("AMZN")

    def test_br_download(self):
        if "BR" in filtered_tickers:
            self.validate_all_data("BR")

    def test_all_tickers_download_and_valid(self):
        errors = {}
        for t in filtered_tickers:
            try:
                self.validate_all_data(t)
            except AssertionError as ex:
                errors[t] = ex
        if len(errors):
            raise AssertionError(
                f"Failed {len(errors)}/{len(filtered_tickers)}, {errors.keys()}"
            )
