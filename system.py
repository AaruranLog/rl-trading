#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt
import gym
import tulipy as ti
from sqlalchemy import create_engine
import ast
import re
import warnings
from wsb_pipeline import get_all_embeddings


class TradingEnv(gym.Env):
    INITIAL_BALANCE = 1
    TRANSACTION_COST = 0.01  # per share
    WINDOW_SIZE = 6
#     DELTA_DAY = pd.Timedelta(days=1)
    
    def __init__(self, ticker="AAPL", target_volatility=10, mode="train"):
        self.ticker = ticker
        self.window = pd.Timedelta(days=self.WINDOW_SIZE)
        assert mode in set(
            ["train", "validation", "test", "dev"]
        ), f"Invalid environment  mode: {mode}"
        self.mode = mode
        self.target_volatility = target_volatility
        self.returns_list = []
        self.rewards_list = []
        self.actions_list = []
        self.values = [self.INITIAL_BALANCE]
        self.cumulative_costs = [0]

        self._compute_simple_states()

    def _compute_simple_states(self):
        self.short_time = 63
        self.long_time = 252
        start, end = self.get_time_endpoints(self.mode)
        self.start = start
        self.end = end

        # 81 needs to be added for some reason to make sure MACD is a number ???
        unexplained = 81
        #         if unexplained:
        #             warn("Using unexplained extra pre-padding.")
        prepadding = pd.Timedelta(
            days=self.short_time + self.long_time + self.WINDOW_SIZE + 1 + unexplained
        )
        postpadding = self.window
        self.prices = data.DataReader(
            self.ticker, "yahoo", start=start - prepadding, end=end + postpadding
        )["Close"]

        # We compute the mean, and standard deviation of the first WINDOW_SIZE days, and use this to standardize
        # the entire time series.
        assert (
            self.WINDOW_SIZE > 1
        ), "WINDOW_SIZE is too small for rolling computations to be meaningful"
        self.mu_hat = self.prices[:self.WINDOW_SIZE].mean()
        self.sigma_hat = self.prices[:self.WINDOW_SIZE].std()

        self.data = pd.DataFrame({"x": (self.prices - self.mu_hat) / self.sigma_hat})
        self.data["logx"] = np.log(self.prices)

        self.data["std"] = self.data["x"].rolling(self.WINDOW_SIZE).std()
        # Use additive returns, because the reward is computed using the additive return
        rets = self.prices - self.prices.shift(1)

        self.data["sharpe"] = (
            rets.rolling(self.WINDOW_SIZE).mean() / rets.rolling(self.WINDOW_SIZE).std()
        )

        exp_short = self.prices.ewm(span=self.short_time, adjust=False).mean()
        exp_long = self.prices.ewm(span=self.long_time, adjust=False).mean()
        self.data["q"] = (
            exp_short - exp_long
        )  # / self.prices.rolling(self.short_time).std()

        macd = ti.macd(
            self.data["x"].values,
            short_period=self.short_time,
            long_period=self.long_time,
            signal_period=self.WINDOW_SIZE,
        )

        self.data["macd_0"] = self.data["macd_1"] = self.data["macd_2"] = np.nan
        self.data["macd_0"][self.long_time - 1 :] = macd[0]
        self.data["macd_1"][self.long_time - 1 :] = macd[1]
        self.data["macd_2"][self.long_time - 1 :] = macd[2]

        # to look up current price from self.data, irrespective of the date break due to the weekend
        self.df_initial_index = self.data.index.get_loc(self.start)
        self.df_index = self.df_initial_index

    def get_time_endpoints(self, mode):
        """
            Start must be in Monday - Friday (??)
        """
        if mode == "train":
            return pd.Timestamp("2014-01-06"), pd.Timestamp("2017-12-31")
        elif mode == "dev":
            return pd.Timestamp("2014-01-06"), pd.Timestamp("2014-12-28")
        elif mode == "test":
            return pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31")
        else:
            raise NotImplementedError()

    def _get_raw_price(self, diff=0):
        return self.prices[self.df_index + diff]

    def _get_normalized_price(self, diff=0):
        return self.data["x"][self.df_index + diff]

    def _get_current_timestamp(self):
        return self.data.index[self.df_index]

    def _get_melted_technical_indicators(self):
        i = self.df_index
        indicators = self.data[(i - self.WINDOW_SIZE + 1) : i + 1]
        return indicators.values.reshape(-1).tolist()

    def _get_current_state(self):
        return self._get_melted_technical_indicators()

    def _get_date(self, diff=0):
        return self.data.index[self.df_index + diff]

    def reset(self):
        self.df_index = self.df_initial_index
        self.returns_list = []
        self.rewards_list = []
        self.actions_list = []
        self.values = [self.INITIAL_BALANCE]
        self.cumulative_costs = [0]
        return self._get_current_state()

    def _compute_reward_function(self, action):
        next_price = self._get_normalized_price(diff=1)
        price = self._get_normalized_price()
        r = next_price - price
        mu = 1

        sigma = self.data["std"][self.df_index]
        sigma_prev = self.data["std"][self.df_index - 1]
        term1 = action * self.target_volatility * r / sigma
        prev_action = self.actions_list[-1] if len(self.actions_list) > 0 else 0
        term2 = (
            price
            * self.TRANSACTION_COST
            * np.abs(term1 - self.target_volatility * prev_action / sigma_prev)
        )
        R = mu * (term1 - term2)

        # Additive Returns as reward function
        #         if action == 1:
        #             R = r - TRANSACTION_COST
        #         elif action == -1:
        #             R = -r - TRANSACTION_COST
        #         elif action == 0:
        #             R = 0 - TRANSACTION_COST
        #         R = action * r + abs(action - prev_action) * TRANSACTION_COST * price
        return R

    def step(self, action):
        """
            Executes an action in the stock environment, using 
            the discrete action space described in: Deep Reinforcement Learning for Trading
            
            i.e. -1 is maximally short, 0 is no holdings, 1 is maximally long
            Inputs: action (one of {-1,0,1})
            Outputs: a tuple (observation/state, step_reward, is_done, info)
        """
        assert action in [-1, 0, 1], f"Got {action} but expected one of {-1, 0, 1}"

        roi = self._get_new_return(action)
        self.returns_list.append(roi)

        R = self._compute_reward_function(action)
        self.rewards_list.append(R)
        self.actions_list.append(action)
        self.df_index += 1
        return (
            self._get_current_state(),
            R,
            self._get_current_timestamp() > self.end,
            {},
        )

    def seed(self, seed=None):
        return

    def close(self):
        final_index = self.df_index
        dates = self.data.index[self.df_initial_index : final_index]
        tickers = [self.ticker] * len(self.actions_list)
        prices = self.prices[self.df_initial_index : final_index]
        assert len(tickers) == len(dates) == len(prices) == len(self.rewards_list)
        history = pd.DataFrame(
            {
                "date": dates,
                "ticker": tickers,
                "rewards": self.rewards_list,
                "actions": self.actions_list,
                "returns": self.returns_list,
                "prices": prices,
            }
        )
        return history

    def _get_new_return(self, a):
        past_value = self.values[-1]
        current_price = self._get_normalized_price()
        next_price = self._get_normalized_price(diff=1)
        prev_a = self.actions_list[-1] if len(self.actions_list) > 0 else 0

        cost_of_trade = abs(prev_a - a) * self.TRANSACTION_COST * past_value / current_price
        self.cumulative_costs.append(cost_of_trade + self.cumulative_costs[-1])
        change_in_value = (next_price / current_price - 1) * a * past_value
        if a == -1:
            max_return_from_short = past_value
            change_in_value = min([change_in_value, max_return_from_short])
        new_value = past_value + change_in_value
        self.values.append(new_value)
        roi = (
            0
            if (new_value == self.INITIAL_BALANCE)
            else (new_value - self.INITIAL_BALANCE) / self.cumulative_costs[-1]
        )
        
#         if new_value < 5 * self.TRANSACTION_COST:
#             warnings.warn(f"Low portfolio value {new_value}")
#         if new_value <= 0:
#             warnings.warn(f"Agent fucked up; portfolio value is {new_value}.")
            
        return roi


class TradingWithRedditEnv(TradingEnv):
    def __init__(self, ticker="AAPL", target_volatility=10, mode="train"):
        super(TradingWithRedditEnv, self).__init__(
            ticker=ticker, target_volatility=target_volatility, mode=mode
        )
        text = get_all_embeddings(ticker=self.ticker)
        text["date"] = pd.to_datetime(text["date"])
        self.text_embeddings = text
        stocks = self.data[self.df_index :]
        stocks["date"] = stocks.index
        self.embedding_lookup = pd.merge(stocks, text, how="left")[
            ["date", "embeddings"]
        ]

    def _get_current_embeddings(self):
        date = self._get_date()
        daily_data = self.embedding_lookup.loc[self.embedding_lookup.date == date]
        # raw_values = daily_data.embeddings.apply(ast.literal_eval).values
        raw_values = daily_data.embeddings.values
        vectors = []
        for s in raw_values:
            try:
                v = ast.literal_eval(s)
                vectors.append(v)
            except:
                vectors.append((np.zeros(50)))

        return vectors

    def _get_current_state(self):
        melted = self._get_melted_technical_indicators()
        embedded = self._get_current_embeddings()
        return melted, embedded
