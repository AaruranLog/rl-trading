#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append("..")
from trading.system import TradingEnv


# In[1]:


import math
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


# In[2]:


def validate_numeric_list(x):
    assert not any(map(math.isnan, x)), "Found nan in x."
    return True

def basic_loop(t, *args, **kwargs):
    env = TradingEnv(ticker=t, *args, **kwargs)
    state = env.reset()
    assert env.prices[:env.WINDOW_SIZE].std() > 0, "Standard Deviation too low"
    validate_numeric_list(state)
    trading_data = env.data[env.df_initial_index - env.WINDOW_SIZE : env.df_final_index]
    assert_frame_equal(trading_data, trading_data.fillna(np.inf))
    done = False
    while not done:
        next_state, r, done, _ = env.step(0)
    h = env.close()
    assert_frame_equal(h, h.fillna(np.inf))


# In[3]:


basic_loop('AAPL', mode="test")


# In[4]:


clean_stocks = []
failed_to_load_list = []
with open('raw_sp500.tsv', 'r') as src:
    for i, line in enumerate(src.readlines()):
        if i == 0:
            continue
        ticker = line.split()[0]
        ticker = ticker.replace('.', '-')
        try:
            basic_loop(ticker)
            basic_loop(ticker, mode='test')
            clean_stocks.append(ticker)
        except:
            print(f'failed on {ticker}')
            failed_to_load_list.append(ticker)
            


# In[6]:


assert len(clean_stocks) == 505, "clean stocks missing. check golden files"


# In[7]:


len(clean_stocks)


# In[12]:


failed_to_load_list.sort()
clean_stocks.sort()


# In[13]:


with open('blacklist.txt', 'w') as target:
    target.write('\n'.join(failed_to_load_list))


# In[14]:


with open('filtered_tickers.txt', 'w') as target:
    target.write('\n'.join([s for s in clean_stocks]))


# In[ ]:




