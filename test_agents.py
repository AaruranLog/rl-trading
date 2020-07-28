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
    assert not any(torch.isnan(p).any() for p in net.parameters())


def basic_episode_test(agent, **kwargs):
    env = TradingEnv(**kwargs)
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
    def test_q_net(self):
        net = QNetwork()
        x1 = FloatTensor([TradingEnv().reset()])
        qs = net(x1)

        x1[0, 0] = float("nan")
        with pytest.raises(AssertionError):
            net(x1)

    def test_dqn_dev(self):
        a = DQN()
        validate_net(a.model)
        validate_net(a.target)
        basic_episode_test(a, mode="dev")
        validate_net(a.model)
        validate_net(a.target)

    def test_dqn_test(self):
        a = DQN()
        validate_net(a.model)
        validate_net(a.target)
        basic_episode_test(a, mode="test")
        validate_net(a.model)
        validate_net(a.target)

    def test_dqn_train(self):
        a = DQN()
        validate_net(a.model)
        validate_net(a.target)
        basic_episode_test(a, mode="train")
        validate_net(a.model)
        validate_net(a.target)


@pytest.mark.incremental
class TestA2C:
    def test_policy_net(self):
        net = PolicyNetwork()
        x1 = FloatTensor([TradingEnv().reset()])
        pi = net(x1, logits=False).detach()
        assert pi.min() >= 0, f"negative probability in policy: {pi}"
        assert abs(pi.sum() - 1) < 1e-3, f"doesn't sum to 1 {pi}"

        x1[0, 0] = float("nan")
        with pytest.raises(AssertionError):
            net(x1)

    def test_a2c_dev(self):
        a = A2C()
        validate_net(a.model)
        validate_net(a.policy)
        basic_episode_test(a, mode="dev")
        validate_net(a.model)
        validate_net(a.policy)

    def test_a2c_test(self):
        a = A2C()
        validate_net(a.model)
        validate_net(a.policy)
        basic_episode_test(a, mode="test")
        validate_net(a.model)
        validate_net(a.policy)

    def test_a2c_train(self):
        a = A2C()
        validate_net(a.model)
        validate_net(a.policy)
        basic_episode_test(a, mode="train")
        validate_net(a.model)
        validate_net(a.policy)


@pytest.mark.incremental
class TestModelBased:
    def test_reward_model(self):
        m = RewardModel()
        state, text = TradingWithRedditEnv().reset()
        state_tensor = FloatTensor([state])
        text_tensor = FloatTensor(text)
        text_tensor = text_tensor.mean(dim=0, keepdim=True)
        m(state_tensor, text_tensor)
        validate_net(m)

    def test_q_text_model(self):
        m = QWithTextModel()
        state, text = TradingWithRedditEnv().reset()
        state_tensor = FloatTensor([state])
        text_tensor = FloatTensor(text)
        text_tensor = text_tensor.mean(dim=0, keepdim=True)
        m(state_tensor, text_tensor)
        validate_net(m)

    def test_transition_model(self):
        state, text = TradingWithRedditEnv().reset()
        state_tensor = FloatTensor([state])
        T = TransitionModel()
        T(state_tensor), T.next_state(state_tensor)

    def test_model_based(self):
        model_agent = ModelBasedAgent()
        e = TradingWithRedditEnv()
        model_agent.run_episode(e)

    def test_model_without_text(self):
        model_without_text = ModelBased_NoText_Agent()
        e_notext = TradingEnv()
        model_without_text.run_episode(e_notext)
