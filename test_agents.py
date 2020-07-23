from agents import *
from system import *
import pytest


def test_base():
    with pytest.raises(AssertionError):
        BaseAgent(gamma=1.1)

    a = BaseAgent()
    assert len(a.filtered_tickers), "No tickers found"


def test_base_convert_action():
    a = BaseAgent()
    env = TradingEnv()

    for i in range(3):
        action_tensor = FloatTensor([[i]])
        assert a.convert_action(action_tensor) == i - 1, "Incorrect conversion"


def validate_net(net):
    if (
        torch.isnan(net.l1.weight).any()
        or torch.isnan(net.l2.weight).any()
        or torch.isnan(net.l3.weight).any()
    ):
        raise ValueError("Weight of net has become NaN")


def test_q_net():
    net = QNetwork()
    x1 = FloatTensor([TradingEnv().reset()])
    qs = net(x1)

    x1[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        net(x1)


def test_policy_net():
    net = PolicyNetwork()
    x1 = FloatTensor([TradingEnv().reset()])
    pi = net(x1, logits=False).detach()
    assert pi.min() >= 0, f"negative probability in policy: {pi}"
    assert pi.sum() == 1, f"doesn't sum to 1 {pi}"

    x1[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        net(x1)


def basic_episode_test(agent):
    env = TradingEnv()
    agent.run_episode(env)


def test_base_basic():
    with pytest.raises(NotImplementedError):
        basic_episode_test(BaseAgent())


def three_tickers_test(agent):
    agent.train(num_tickers=3, num_episodes=3)


def test_longonly_basic():
    basic_episode_test(LongOnlyAgent())
    long_agent = LongOnlyAgent()
    long_agent.train(num_tickers=1, num_episodes=100)


def test_longonly_3tickers():
    three_tickers_test(LongOnlyAgent())


@pytest.mark.incremental
class TestDQN:
    def test_dqn_basic(self):
        basic_episode_test(DQN())

    def test_dqn_3tickers(self):
        a = DQN()
        three_tickers_test(a)
        validate_net(a.model)
        validate_net(a.target)

    def test_dqn_long(self):
        dqn_agent = DQN()
        dqn_agent.train(env_mode='train', num_tickers=10, num_episodes=10)
        
    def test_dqn_long(self):
        dqn_agent = DQN()
        dqn_agent.train(env_mode='train', num_tickers=10, num_episodes=10)


@pytest.mark.incremental
class TestA2C:
    def test_a2c_basic(self):
        basic_episode_test(A2C())

    def test_a2c_3tickers(self):
        a = A2C()
        three_tickers_test(a)
        validate_net(a.model)
        validate_net(a.policy)

    def test_a2c_large(self):
        a2c_agent = A2C()
        a2c_agent.train(env_mode="train", num_tickers=10, num_episodes=10)
        validate_net(a2c_agent.model)
        validate_net(a2c_agent.policy)

    def test_a2c_all(self):
        pass
