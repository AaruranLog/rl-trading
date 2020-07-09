#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt
import gym
from warnings import warn

INITIAL_BALANCE = 10
TRANSACTION_COST = 0.01
WINDOW_SIZE = 6
DELTA_DAY = pd.Timedelta(days=1)
DEFAULT_ACTIONS_LIST = [0]
DEFAULT_REWARDS_LIST = [0]
EXP_DECAY = 0.8
class TradingEnv(gym.Env):
    def __init__(self, ticker='AAPL', target_volatility=1, mode="train"):
        self.ticker = ticker
        self.window = pd.Timedelta(days=WINDOW_SIZE)
        assert mode in set(["train", "validation", "test", "dev"]), f"Invalid environment  mode: {mode}"
        self.mode = mode
       
        self.target_volatility = target_volatility
        self.returns_list = DEFAULT_REWARDS_LIST.copy()
        self.rewards_list = DEFAULT_REWARDS_LIST.copy()
        self.actions_list = DEFAULT_ACTIONS_LIST.copy()
        self.balance = INITIAL_BALANCE
        
        self._compute_technical_indicators()
        
    def _compute_technical_indicators(self):
        self.short_time = 63
        self.long_time = 252
        start, end = self.get_time_endpoints(self.mode)
        self.start = start
        self.end = end
        prepadding =  pd.Timedelta(days=max([self.short_time + self.long_time, WINDOW_SIZE]) + 1) 
        postpadding = self.window
        self.prices = data.DataReader(self.ticker, 'yahoo',
                                      start=start-prepadding, end=end+postpadding)['Close']

        # We compute the mean, and standard deviation of the first WINDOW_SIZE days, and use this to standardize 
        # the entire time series.
        self.mu_hat = self.prices[:WINDOW_SIZE].mean()
        self.sigma_hat = self.prices[:WINDOW_SIZE].std()
        self.data = pd.DataFrame({'mean' : (self.prices - self.mu_hat) / self.sigma_hat})
        self.data['std'] = self.data['mean'].rolling(WINDOW_SIZE).std()
        # Use additive returns, because the reward is computed using the additive return
        rets = (self.prices - self.prices.shift(1))

        self.data['sharpe'] = rets.rolling(WINDOW_SIZE).mean() / rets.rolling(WINDOW_SIZE).std()
#         warn('Sharpe ratio will need a risk-free return in the future, for proper calculation.')
        
        exp_short = self.prices.ewm(span=self.short_time, adjust=False).mean() # ???
        exp_long  = self.prices.ewm(span=self.long_time,  adjust=False).mean()  # ???
        self.data['q'] = (exp_short - exp_long) / self.prices.rolling(self.short_time).std()
        self.data['MACD'] = self.data['q'] / self.data['q'].rolling(self.long_time).std()
        
        # to look up current price from self.data, irrespective of the date break due to the weekend
        self.df_index = self.data.index.get_loc(self.start)
        
        
    def get_time_endpoints(self, mode):
        """
            Start must be in Monday - Friday
        """
        if mode == "train":
            return pd.Timestamp('2016-01-04'), pd.Timestamp('2018-12-31')
        elif mode == "dev":
            return pd.Timestamp('2016-01-04'), pd.Timestamp('2016-02-28')
        else:
            raise NotImplementedError()
        
    def _get_raw_price(self):
        return self.prices[self.df_index]
    
    def _get_normalized_price(self, diff=0):
        return self.data['mean'][self.df_index + diff]
        
    def _get_current_timestamp(self):
        return self.data.index[self.df_index]
    
    def _get_current_state(self):
        state = []
        for i in range(WINDOW_SIZE):
            n_price = self._get_normalized_price(diff=-i)
            state.append(n_price)
            
            old_price = self._get_normalized_price(diff=-(i + WINDOW_SIZE))
            state.append(old_price)
            
            sharpe_ratio = self.data['sharpe'][self.df_index - i]
            state.append(sharpe_ratio)
            
            # Normalized, Additive Returns from previous WINDOW_SIZE
            state.append(n_price - old_price)
        return state
    
    def reset(self):
        self.df_index = self.data.index.get_loc(self.start)  
        self.returns_list = DEFAULT_REWARDS_LIST.copy()
        self.rewards_list = DEFAULT_REWARDS_LIST.copy()
        self.actions_list = DEFAULT_ACTIONS_LIST.copy()
        return self._get_current_state()

                
    def step(self, action):
        """
            Executes an action in the stock environment, using 
            the discrete action space described in: Deep Reinforcement Learning for Trading
            
            i.e. -1 is maximally short, 0 is no holdings, 1 is maximally long
            Inputs: action (one of {-1,0,1})
            Outputs: a tuple (observation/state, step_reward, is_done, info)
        """
        assert action in [-1, 0, 1], f"Got {action} but expected one of {-1, 0, 1}"
        next_price = self._get_normalized_price(diff=1)
        price = self._get_normalized_price()
        r = next_price - price
        mu = 1
        
        sigma = self.data['std'][self.df_index - 1] 
        sigma_prev = self.data['std'][self.df_index - 2]
       
        term1 = action * self.target_volatility * r / sigma
        prev_action = self.actions_list[-1]
        term2 = price * TRANSACTION_COST * np.abs(term1 - self.target_volatility * prev_action / sigma_prev)
        R = mu*(term1 - term2)
        
        # TODO: Refactor rewards_list, actions_list into a pd.DataFrame so that
        # 1. I can plot things more easily, and group them together by ticker, and episode number
        # 2. I can collect rewards_list, actions_list into a single variable
        
        self.rewards_list.append(R)
        self.actions_list.append(action)
        self.df_index += 1
        return self._get_current_state(), R, self._get_current_timestamp() > self.end, {}

        
    def seed(self, seed=None):
        return
    
    def close(self):
        return
    
    def _update_portfolio(action):
        raise NotImplementedError()
        prev_action = self.actions_list[-1]
        if prev_action == 0:
            if action == 0:
                return self.balance
            elif action == 1:
                return self.balance
            elif action == -1:
                return self.balance
        if prev_action == 1:
            if action == 0:
                return self.balance
            elif action == 1:
                return self.balance
            elif action == -1:
                pass
        if prev_action == -1:
            if action == 0:
                return self.balance
            elif action == 1:
                return self.balance
            elif action == -1:
                pass