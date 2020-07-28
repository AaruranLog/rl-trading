#!/usr/bin/env python
# coding: utf-8

from system import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import numpy as np
import pandas as pd
import random
import math
from tqdm import tqdm


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


STATE_DIM = len(TradingEnv().reset())
#     print(f"STATE_DIM = {STATE_DIM}")
EMBED_DIM = 50  # from the dimensionality-reduced fastText model
HIDDEN_LAYER = 70  # NN hidden layer size
ACTION_DIM = 3

# EPISODES = 2000  # number of episodes
# EPS_START = 0.9  # e-greedy threshold start value
# EPS_END = 0.05  # e-greedy threshold end value
# EPS_DECAY = 200  # e-greedy threshold decay
# # GAMMA = 0.99  # Q-learning discount factor
# LR = 0.001  # NN optimizer learning rate
# HIDDEN_LAYER = 128  # NN hidden layer size
# BATCH_SIZE = 16  # Q-learning batch size
# TARGET_UPDATE = 100  # frequency of target update
# BUFFER_SIZE = 100  # capacity of the replay buffer

# if gpu is to be used
# use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


if use_cuda:
    print("GPU found and in use")
else:
    print("No GPU will be used")


class QNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(STATE_DIM, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, ACTION_DIM)

    def forward(self, x):
        """
            Computes the estimated Q-values for a given batch x
        """
        assert not torch.isnan(x).any(), f"NaN in input {x}"
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def sample_from_softmax_policy(self, batch_state):
        """
            Maps a batch of states into actions in {-1,0,1}
        """
        batch_q_values = self.forward(batch_state)
        batch_pi = F.softmax(batch_q_values, dim=1)
        batch_size = batch_pi.shape[0]
        batch_actions = torch.empty(batch_size, 1)
        for i in range(batch_size):
            pi = batch_pi[i, :]
            dist = torch.distributions.Categorical(pi)
            # Subtract 1, so batch_actions is in {-1, 0, 1}
            batch_actions[i, 0] = dist.sample().view(1, 1) - 1
        if use_cuda:
            batch_actions = batch_actions.to(batch_state.get_device())
        return batch_actions.long()


class BaseAgent:
    #     EPISODES = 2000  # number of episodes

    LR = 0.001  # NN optimizer learning rate

    BATCH_SIZE = 16  # Q-learning batch size
    TARGET_UPDATE = 100  # frequency of target update
    BUFFER_SIZE = 100  # capacity of the replay buffer
    ENV_CONSTRUCTOR = TradingEnv
    def __init__(self, gamma=0.8):
        assert 0 < gamma < 1, f"Invalid gamma: {gamma}"
        sns.set()
        self.gamma = gamma
        self.memory = ReplayMemory(self.BUFFER_SIZE)
        self.history = pd.DataFrame()
        self.name = ""
        self.steps_done = 0
        with open("filtered_tickers.txt", "r") as src:
            self.filtered_tickers = src.read().split("\n")

    def run_episode(self, environment):
        """
            Takes an env, and trains the agent until the environment
            reaches a terminal state (ie the training window is complete).
            Must return environment.close()
        """
        raise NotImplementedError()

    def plot_cumulative_discounted_rewards(self):
        raise NotImplementedError("Needs to be refactored.")
        rl_data = self.history
        rl_data["discount_factor"] = np.power(self.gamma, rl_data.episode - 1)
        rl_data["discounted_future_reward"] = (
            rl_data["discount_factor"] * rl_data["rewards"]
        )
        rl_data = rl_data[["episode", "discounted_future_reward"]]
        rl_data = rl_data.groupby("episode").sum()
        #         rl_plot = sns.lineplot(data=rl_data, legend=False)
        title = "Cumulative Discounted Rewards over Episodes"
        if len(self.name):
            title = f"Cumulative Discounted Reward for {self.name}"

        rl_data.plot(legend=False, title=title)
        plt.ylabel("Cumulative Discounted Reward")
        if len(self.name):
            filename = f"plots/rewards-{self.name}.png"
            plt.savefig(filename)
        plt.show()

    def convert_action(self, action):
        assert action in [0, 1, 2], f"Invalid action: {action}"
        position = action - 1
        #         assert position in [-1,0,1]
        return position.item()

    def train(self, num_tickers=20, episodes_per_ticker=5, **kwargs):
        """
            Trains the agent for episode_per_ticker, on each of num_tickers, looping over the approved
            list of tickers. This is a convenience function.
        """
        num_tickers = min(num_tickers, len(self.filtered_tickers))
        for i in range(num_tickers):
            ticker = self.filtered_tickers[i % num_tickers]
            env = self.ENV_CONSTRUCTOR(ticker=ticker, **kwargs)
            for j in tqdm(range(episodes_per_ticker)):
                history = self.run_episode(env)
                history["ticker"] = ticker
                history["episode"] = j + 1
                self.history = pd.concat((self.history, history))
        self.history = self.history.reset_index("Date", drop=True)


    def plot_returns(self, ticker):
        h = self.history
        roi_data = h[h.ticker == ticker][["date", "episode", "returns"]]
        plt.title(f"Returns for {ticker}")
        roi_plot = sns.lineplot(data=h, x="date", y="returns", hue="episode")
        roi_plot.set_xticklabels(roi_plot.get_xticklabels(), rotation=45)


class DQN(BaseAgent):
    EPS_START = 0.9  # e-greedy threshold start value
    EPS_END = 0.05  # e-greedy threshold end value
    EPS_DECAY = 200  # e-greedy threshold decay
    LR = 0.001  # NN optimizer learning rate

    def __init__(self):
        super().__init__()
        self.model = QNetwork()
        self.target = QNetwork()
        if use_cuda:
            self.model.cuda()
            self.target.cuda()
        self.name = "Deep Q-Network"
        self.optimizer = optim.SGD(self.model.parameters(), self.LR)

    def select_epsilon_greedy_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        greedy_action = None
        with torch.no_grad():
            greedy_action = self.model(state).data.argmax(dim=1).view(1, 1)

        random_action = LongTensor([[random.randrange(ACTION_DIM)]])
        assert (
            greedy_action.shape == random_action.shape
        ), f"Incorrect sampling techinque {greedy_action.shape, random_action.shape}"
        if sample > eps_threshold:
            return greedy_action
        else:
            return random_action

    def run_episode(self, environment):
        state = environment.reset()
        steps = 0
        action = None
        while True:
            state_tensor = FloatTensor([state])
            action = self.select_epsilon_greedy_action(state_tensor)
            position = self.convert_action(action)
            next_state, reward, done, _ = environment.step(position)

            self.memory.push(
                (
                    FloatTensor([state]),
                    action,  # action is already a tensor
                    FloatTensor([next_state]),
                    FloatTensor([reward]),
                    FloatTensor([int(done)]),
                )
            )

            self.learn()
            if self.steps_done % self.TARGET_UPDATE == 0:
                self.target.load_state_dict(self.model.state_dict())

            state = next_state
            steps += 1
            if done:
                break
        history = environment.close()
        return history

    def max_next_q_values(self, batch_next_state):
        # expected Q values are estimated from actions which gives maximum Q value
        return self.target(batch_next_state).detach().max(1)[0]

    def learn(self):
        if len(self.memory) <= self.BATCH_SIZE:
            return
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(
            *transitions
        )
        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state))
        batch_done = Variable(torch.cat(batch_done))

        # current Q values are estimated by NN for all actions
        current_q_values = self.model(batch_state).gather(1, batch_action).squeeze()
        expected_future_rewards = self.max_next_q_values(batch_next_state)

        expected_q_values = batch_reward + (self.gamma * expected_future_rewards) * (
            1 - batch_done
        )

        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # backpropagation of loss to QNetwork
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class LongOnlyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "Long-Only Strategy"

    def run_episode(self, environment):
        state = environment.reset()
        position = 1
        done = False
        while not done:
            _, r, done, __ = environment.step(position)
            self.steps_done += 1

        return environment.close()
    
    def train(self, num_tickers=20, episodes_per_ticker=5, **kwargs):
        """
            Trains the agent for episode_per_ticker, on each of num_tickers, looping over the approved
            list of tickers. This is a convenience function.
        """
        num_tickers = min(num_tickers, len(self.filtered_tickers))
        for i in range(num_tickers):
            ticker = self.filtered_tickers[i % num_tickers]
            env = self.ENV_CONSTRUCTOR(ticker=ticker, **kwargs)
            for j in tqdm(range(episodes_per_ticker)):
                history = self.run_episode(env)
                history["ticker"] = ticker
                history["episode"] = j + 1
                self.history = pd.concat((self.history, history))
        self.history = self.history.reset_index("Date", drop=True)


class PolicyNetwork(nn.Module):
    # for Policy-Gradient methods, e.g. actor-only and actor-critic methods
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(STATE_DIM, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, ACTION_DIM)  # 2, for the action

    def forward(self, x, logits=True):
        """
            returns the logits of the probability
        """
        assert not torch.isnan(x).any(), f"NaN in input {x}"
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        if not logits:
            x = torch.softmax(x, dim=1)
        return x

    def sample_from_softmax_policy(self, batch_state):
        batch_logits = self.forward(batch_state).detach()
        assert not torch.isnan(
            batch_logits
        ).any(), f"NaN in policy logits {batch_logits}"
        batch_size = batch_logits.shape[0]
        actions = torch.empty(batch_size, 1)
        for i in range(batch_size):
            logits = batch_logits[i, :]
            dist = torch.distributions.Categorical(logits=logits)
            actions[i, 0] = dist.sample().view(1, 1)
        if use_cuda:
            actions = actions.to(batch_state.get_device())
        return actions.long()


class A2C(BaseAgent):
    def __init__(self):
        super().__init__()
        self.name = "A2C"
        self.policy = PolicyNetwork()
        self.model = QNetwork()
        if use_cuda:
            self.policy.cuda()
            self.model.cuda()

        self.optimizer = optim.Adam(
            chain(self.model.parameters(), self.policy.parameters()), self.LR
        )

    def run_episode(self, environment):
        state = environment.reset()
        self.steps_done = 0
        action = None
        while True:
            state_tensor = FloatTensor([state])
            action = self.policy.sample_from_softmax_policy(state_tensor)
            position = self.convert_action(action)
            next_state, reward, done, _ = environment.step(position)
            next_state_tensor = FloatTensor([next_state])
            self.learn(state_tensor, action, next_state_tensor, reward)
            state = next_state
            self.steps_done += 1
            if done:
                break
        history = environment.close()
        return history

    def learn(self, state_tensor, action, next_state_tensor, reward):
        n = self.steps_done
        expected_q = reward + self.gamma * self.model(next_state_tensor).max(dim=1)[0]
        q_values = self.model(state_tensor)
        current_q = q_values.gather(1, action).squeeze(0)
        q_loss = F.smooth_l1_loss(current_q, expected_q.detach())

        pi = self.policy(state_tensor, logits=False)
        A = expected_q - torch.dot(q_values.squeeze(0), pi.squeeze(0))
        pi_a = pi.gather(1, action)
        policy_loss = -A.detach() * torch.log(pi_a)

        loss = (self.gamma ** n) * (policy_loss + q_loss)

        # backpropagation of loss to QNetwork and PolicyNetwork
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
class DeterministicPolicyNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(STATE_DIM, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l3 = nn.Linear(HIDDEN_LAYER, 1)

    def forward(self, x):
        """
            returns the position (in [-1, 1])
        """
        assert not torch.isnan(x).any(), f"NaN in input {x}"
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return torch.clamp(x, -1, 1)
    
    
class DeterministicQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_s = nn.Linear(STATE_DIM, HIDDEN_LAYER)
        self.l1_p = nn.Linear(1, HIDDEN_LAYER)
        self.l2 = nn.Linear(2 * HIDDEN_LAYER, 2 * HIDDEN_LAYER)
        self.l3 = nn.Linear(2 * HIDDEN_LAYER, 1)
        
    def forward(self, s, p):
        """
            maps (state, position) to a q-value
        """
        h1 = F.relu(self.l1_s(s))
        h2 = F.relu(self.l1_p(p))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.l2(cat))
        q = self.l3(q)
        return q    

    
class DDPG(BaseAgent):
    ENV_CONSTRUCTOR = ContinuousTradingEnv
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DDPG"
        self.policy = DeterministicPolicyNetwork()
        self.policy_opt = optim.Adam(self.policy.parameters())
        
        self.model = DeterministicQNetwork()
        self.model_opt = optim.Adam(self.model.parameters())
        
    def run_episode(self, environment):
        state = environment.reset()
        self.steps_done = 0
        while True:
            state_tensor = FloatTensor([state])
            assert not torch.isnan(state_tensor).any()
            posn = self.policy(state_tensor)
            next_state, reward, done, _ = environment.step(posn.item())
            next_state_tensor = FloatTensor([next_state])
            self.learn(state_tensor, posn, next_state_tensor, reward)
            state = next_state
            self.steps_done += 1
            if done:
                break
        history = environment.close()
        return history
    
    def learn(self, state_tensor, posn, next_state_tensor, reward):
        n = self.steps_done
        next_posn = self.policy(next_state_tensor)
        expected_q = reward + self.gamma * self.model(next_state_tensor, next_posn)
        q_values = self.model(state_tensor, posn)
        current_q = q_values

        q_loss = F.smooth_l1_loss(current_q, expected_q.detach()) * (self.gamma ** n)
        
        self.model_opt.zero_grad()
        q_loss.backward()
        self.model_opt.step()
        
        suggested_posn = self.policy(state_tensor)
        policy_loss = -self.model(state_tensor, suggested_posn) * (self.gamma ** n)
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        
    def train(self, env_mode="train", num_tickers=20, num_episodes=5, **kwargs):
        """
            Trains the agent for num_episodes episodes, looping over the approved
            list of tickers (filtered by num_tickers). This is a convenience function.
        """
        num_tickers = min(num_tickers, len(self.filtered_tickers))
        if num_tickers == np.Inf:
            num_tickers = len(self.filtered_tickers)
        self.history = pd.DataFrame()
        for i in tqdm(range(num_episodes)):
            ticker = self.filtered_tickers[i % num_tickers]
            env = ContinuousTradingEnv(ticker=ticker, **kwargs)
            history = self.run_episode(env)
            history["ticker"] = ticker
            history["episode"] = i + 1
            self.history = pd.concat((self.history, history))
        self.history = self.history.reset_index("Date", drop=True)

def main():
    with open("filtered_tickers.txt", "r") as src:
        filtered_tickers = src.read().split("\n")

    agent_constructors = [DQN, A2C]
    training_history = pd.DataFrame()
    for con in agent_constructors:
        for t in tqdm(filtered_tickers):
            a = con()
            e = a.ENV_CONSTRUCTOR(ticker=t, mode="dev")
            h = a.run_episode(e)
            h['agent'] = a.name
            h['t'] = range(len(h))
            training_history = pd.concat((training_history, h))

#         a.plot_cumulative_discounted_rewards()
        
if __name__ == "__main__":
    main()

# a2c_agent.plot_returns("MMM")

# # In[ ]:


# q = sns.FacetGrid(
#     a2c_agent.history[a2c_agent.history.episode % 10 == 0],
#     col="episode",
#     col_wrap=3,
#     aspect=1.61,
# )
# q.map(sns.lineplot, "date", "returns")


# # In[ ]:


# long_agent.plot_cumulative_discounted_rewards()


# # In[ ]:


# a2c_agent.plot_returns("MMM")


# # In[34]:


# h = a2c_agent.history
# q = sns.FacetGrid(h[h.episode % 20 == 0], col="episode", row="ticker")
# q.map(sns.lineplot, "date", "returns")


# # In[ ]:


class ModelBasedAgent(BaseAgent):
    def __init__(self):
        super().__init__()
