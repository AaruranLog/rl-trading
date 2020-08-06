#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import gym
import tulipy as ti
from sqlalchemy import create_engine
import ast
import re
import warnings
from trading.wsb_pipeline import get_all_embeddings
import requests_cache
import datetime
import math

pd.options.mode.chained_assignment = None


class TradingEnv(gym.Env):
    INITIAL_BALANCE = 10
    TRANSACTION_COST = 0.0001  # per share
    WINDOW_SIZE = 14
    expire_after = datetime.timedelta(days=14)
    session = requests_cache.CachedSession(
        cache_name="cache", backend="sqlite", expire_after=expire_after
    )

    def __init__(
        self, ticker="AAPL", target_volatility=1, mode="train", window_size=14
    ):
        self.ticker = ticker
        self.WINDOW_SIZE = window_size
        self.window = pd.Timedelta(days=self.WINDOW_SIZE)
        assert mode in set(
            ["train", "validation", "test", "dev"]
        ), f"Invalid environment  mode: {mode}"
        self.mode = mode
        self.target_volatility = target_volatility
        self.returns_list = []
        self.rewards_list = []
        self.actions_list = []

        self.cash = [self.INITIAL_BALANCE]
        self.investment_value = [0]
        self.cumulative_costs = [0]
        self._compute_simple_states()

    def _compute_simple_states(self):
        self.short_time = 12
        self.long_time = 26
        start, end = self.get_time_endpoints(self.mode)
        self.start = start
        self.end = end
        # We need prepadding for MACD, and the rolling calcuations
        prepadding = pd.Timedelta(
            days=self.short_time + self.long_time + self.WINDOW_SIZE + 7
        )

        postpadding = pd.Timedelta(
            days=7
        )  # to get around the weekend and possible holidays
        self.prices = web.DataReader(
            self.ticker,
            "yahoo",
            start=start - prepadding,
            end=end + postpadding,
            session=self.session,
        )["Close"]
        self.prices_pct_change = self.prices.pct_change()

        # We must rescale the data dynamically for numerical stability.
        min_prices = self.prices.expanding().min()
        max_prices = self.prices.expanding().max()
        assert not all(min_prices == max_prices), "Price doesn't change"
        width = max_prices - min_prices
        width[0] = width[1]
        center = self.prices.expanding().median()
        scaled_prices = (self.prices - center) / width
        self.prices_normalized = scaled_prices
        # We compute the mean, and standard deviation of the first WINDOW_SIZE days, and use this to standardize
        # the entire time series.
        assert self.WINDOW_SIZE > 1, "WINDOW_SIZE is too small"

        self.data = pd.DataFrame()

        self.data["x"] = scaled_prices
        self.data["diff_x"] = self.data["x"].diff(-1)

        self.mu_hat = self.data["x"][: self.WINDOW_SIZE].mean()
        self.sigma_hat = self.data["x"][: self.WINDOW_SIZE].std()

        self.data["std"] = self.data["x"].rolling(self.WINDOW_SIZE).std()
        smallest_nonzero_std = self.data["std"][self.data["std"] > 0].expanding().min()
        self.data["std"][self.data["std"] == 0] = smallest_nonzero_std[
            self.data["std"] == 0
        ]
        # Use additive returns, because the reward is computed using the additive return
        #         rets = self.prices - self.prices.shift(-1)
        rets = self.prices.diff().shift(-1)

        self.data["sharpe"] = (
            rets.rolling(self.WINDOW_SIZE).mean() / rets.rolling(self.WINDOW_SIZE).std()
        )
        self.data["sharpe"][self.data["sharpe"].apply(math.isnan)] = 0

        macd = ti.macd(
            self.prices.values,
            short_period=self.short_time,
            long_period=self.long_time,
            signal_period=self.WINDOW_SIZE,
        )

        self.data["macd"] = 0
        self.data["macd"][self.long_time - 1 :] = macd[2]
        # to look up current price from self.data, irrespective of the date break due to the weekend
        self.df_initial_index = self.data.index.get_loc(self.start)
        self.df_index = self.df_initial_index
        self.df_final_index = self.data.index.get_loc(self.end)

    def get_time_endpoints(self, mode):
        """
            Start must be in Monday - Friday (??)
        """
        if mode == "train":
            return pd.Timestamp("2014-01-06"), pd.Timestamp("2017-12-29")
        elif mode == "dev":
            return pd.Timestamp("2014-01-06"), pd.Timestamp("2014-12-31")
        elif mode == "test":
            return pd.Timestamp("2018-01-02"), pd.Timestamp("2018-12-31")
        else:
            raise ValueError(f"Invalide mode = {mode}")

    def _get_raw_price(self, diff=0):
        return self.prices[self.df_index + diff]

    def _get_normalized_price(self, diff=0):
        return self.prices_normalized[self.df_index + diff]

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
        self.cash = [self.INITIAL_BALANCE]
        self.investment_value = [0]
        self.cumulative_costs = [0]
        return self._get_current_state()

    def _compute_reward_function(self, action):
        #         next_price = np.log(self._get_normalized_price(diff=1))
        #         price = np.log(self._get_normalized_price())

        new_value = (
            self.cash[-1] + self.investment_value[-1] - self.cumulative_costs[-1]
        )
        prev_value = (
            self.cash[-2] + self.investment_value[-2] - self.cumulative_costs[-2]
        )
        R = new_value - prev_value

        #         next_price = self._get_normalized_price(diff=1)
        #         price = self._get_normalized_price()
        #         r = next_price - price
        #         mu = 1

        #         sigma = self.data["std"][self.df_index]
        #         sigma_prev = self.data["std"][self.df_index - 1]
        #         term1 = action * self.target_volatility * r / sigma
        #         prev_action = self.actions_list[-1] if len(self.actions_list) > 0 else 0
        #         term2 = (
        #             price
        #             * self.TRANSACTION_COST
        #             * np.abs(term1 - self.target_volatility * prev_action / sigma_prev)
        #         )
        #         R = mu * (term1 - term2)
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
        prev_roi = 0 if len(self.returns_list) == 0 else self.returns_list[-1]
        roi = self._get_new_return(action)
        self.returns_list.append(roi)

        R = self._compute_reward_function(action)
        # #         R = (roi + prev_roi) / (np.sqrt(2) * np.abs(roi - prev_roi) + 1e-6)
        # #         R = roi - prev_roi
        #         new_value = self.cash[-1] + self.investment_value[-1] - self.cumulative_costs[-1]
        #         prev_value= self.cash[-2] + self.investment_value[-2] - self.cumulative_costs[-2]
        #         R = (new_value - prev_value)
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
        self._update_values(a)
        latest_value = self.cash[-1] + self.investment_value[-1]
        initial_value = self.INITIAL_BALANCE
        total_cost = self.cumulative_costs[-1]
        roi = (
            0
            if (latest_value == self.INITIAL_BALANCE)
            else (latest_value - self.INITIAL_BALANCE) / total_cost
        )
        return roi

    def _update_values(self, a):
        current_value = self.cash[-1] + self.investment_value[-1]
        preserved_value = current_value * (1 - np.abs(a))  # new cash balance
        invested_value = current_value * np.abs(a)
        change_in_invested_value = (
            self.prices_pct_change[self.df_index] * np.sign(a) * invested_value
        )
        if a < 0:
            change_in_invested_value = min(
                change_in_invested_value, invested_value
            )  # max return from a short sale
        new_invested_value = invested_value + change_in_invested_value

        # no additional cost if the position is constant,
        # cost of admission (ie paying the market) and cost of investment (i.e. the lost cash)
        prev_a = self.actions_list[-1] if len(self.actions_list) else 0
        cost_of_trade = np.abs(prev_a - a) * current_value * (1 + self.TRANSACTION_COST)

        self.cash.append(preserved_value)
        self.cumulative_costs.append(self.cumulative_costs[-1] + cost_of_trade)
        self.investment_value.append(new_invested_value)


class TradingWithRedditEnv(TradingEnv):
    def __init__(self, ticker="AAPL", target_volatility=10, mode="train"):
        super(TradingWithRedditEnv, self).__init__(
            ticker=ticker, target_volatility=target_volatility, mode=mode
        )
        text = get_all_embeddings(ticker=self.ticker)
        text["date"] = pd.to_datetime(text["date"])
        self.text_embeddings = text
        stocks = self.data[self.df_index :]
        #         stocks["date"] = stocks.index
        stocks = stocks.reset_index("Date")
        stocks["date"] = stocks["Date"]
        #         stocks.drop('Date', inplace=True)
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


class ContinuousTradingEnv(TradingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        """
            Executes an action in the stock environment, using 
            the CONTINUOUS action space described in: Deep Reinforcement Learning for Trading
            
            i.e. -1 is maximally short, 0 is no holdings, 1 is maximally long
            Inputs: action in [-1, 1]
            Outputs: a tuple (observation/state, step_reward, is_done, info)
        """
        assert -1 <= action <= 1, f"Got {action} but it is outside of [-1, 1]"

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
