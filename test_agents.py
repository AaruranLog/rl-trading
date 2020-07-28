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
    assert abs(pi.sum() - 1) < 1e-3, f"doesn't sum to 1 {pi}"

    x1[0, 0] = float("nan")
    with pytest.raises(AssertionError):
        net(x1)


def basic_episode_test(agent):
    env = TradingEnv()
    agent.run_episode(env)


def test_base_basic():
    basic_episode_test(BaseAgent())


def three_tickers_test(agent):
    agent.train(num_tickers=3, episodes_per_ticker=3)


def test_longonly_basic():
    basic_episode_test(LongOnlyAgent())
    long_agent = LongOnlyAgent()
    long_agent.train(num_tickers=1)


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
        dqn_agent.train(num_tickers=10, episodes_per_ticker=10, mode="train")

    def test_dqn_long(self):
        dqn_agent = DQN()
        dqn_agent.train(num_tickers=10, episodes_per_ticker=10, mode="train")


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
        a2c_agent.train(num_tickers=10, episodes_per_ticker=1, mode="train")
        validate_net(a2c_agent.model)
        validate_net(a2c_agent.policy)

    def test_a2c_all(self):
        pass

def test_reward_model():
    m = RewardModel()
    state, text = TradingWithRedditEnv()
    state_tensor = FloatTensor([state])
    text_tensor  = FloatTensor(text)
    text_tensor = text_tensor.mean(dim=0, keepdim=True)
    m(state_tensor, text_tensor)
    validate_net(m)
    
def test_q_text_model():
    m = QWithTextModel()
    state, text = TradingWithRedditEnv()
    state_tensor = FloatTensor([state])
    text_tensor  = FloatTensor(text)
    text_tensor = text_tensor.mean(dim=0, keepdim=True)
    m(state_tensor, text_tensor)
    validate_net(m)
    
def test_transition_model():
    state, text = TradingWithRedditEnv()
    state_tensor = FloatTensor([state])
    T = TransitionModel()
    T(state_tensor), T.next_state(state_tensor)
    
def test_model_based():
    model_agent = ModelBasedAgent()
    e = TradingWithRedditEnv()
    model_agent.run_episode(e)
    
def test_model_without_text():
    model_without_text = ModelBased_NoText_Agent()
    e_notext = TradingEnv()
    model_without_text.run_episode(e_notext)