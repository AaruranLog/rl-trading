from trading.agents import *
from trading.system import *
import pytest


def test_base_convert_action():
    a = BaseAgent()
    env = TradingEnv()

    for i in range(3):
        action_tensor = FloatTensor([[i]])
        assert a.convert_action(action_tensor) == i - 1, "Incorrect conversion"
        
def test_gamma():
    with pytest.raises(AssertionError):
        BaseAgent(gamma=1.1)
        

@pytest.mark.incremental
class TestBaseAgent:
    agent_constructor = BaseAgent
        
    def test_tickers(self):
        a = self.agent_constructor()
        assert len(a.filtered_tickers), "No tickers found"
                
        
    def basic_episode_test(self, agent, **kwargs):
        env = TradingEnv(**kwargs)
        agent.run_episode(env)
        
        
    def test_dev_env(self):
        agent = self.agent_constructor()
        self.basic_episode_test(agent, mode='dev')
        
    def test_test_env(self):
        agent = self.agent_constructor()        
        self.basic_episode_test(agent, mode='test')
    
    def test_train_env(self):
        agent = self.agent_constructor()        
        self.basic_episode_test(agent, mode='train')
        
    def test_train_helper(self):
        agent = self.agent_constructor()
        agent.train(num_tickers=1, episodes_per_ticker=1, mode='dev')
        
    def test_three_tickers(self, **kwargs):
        agent = self.agent_constructor(**kwargs)
        agent.train(num_tickers=3, episodes_per_ticker=1)



def validate_net(net):
    assert not any(torch.isnan(p).any() for p in net.parameters())


class TestLongOnlyAgent(TestBaseAgent):
    agent_constructor = LongOnlyAgent
    

@pytest.mark.incremental
class TestDQN(TestBaseAgent):
    agent_constructor = DQN

    def test_q_net(self):
        net = QNetwork()
        x1 = FloatTensor([TradingEnv().reset()])
        qs = net(x1)

        x1[0, 0] = float("nan")
        with pytest.raises(AssertionError):
            net(x1)
    
    def basic_episode_test(self, agent, **kwargs):
        env = TradingEnv(**kwargs)
        validate_net(agent.model)
        validate_net(agent.target)
        agent.run_episode(env)
        validate_net(agent.model)
        validate_net(agent.target)
            

@pytest.mark.incremental
class TestA2C(TestBaseAgent):
    agent_constructor = A2C
    def test_policy_net(self):
        net = PolicyNetwork()
        x1 = FloatTensor([TradingEnv().reset()])
        pi = net(x1, logits=False).detach()
        assert pi.min() >= 0, f"negative probability in policy: {pi}"
        assert abs(pi.sum() - 1) < 1e-3, f"doesn't sum to 1 {pi}"

        x1[0, 0] = float("nan")
        with pytest.raises(AssertionError):
            net(x1)
    
    def basic_episode_test(self, agent, **kwargs):
        env = TradingEnv(**kwargs)
        validate_net(agent.model)
        validate_net(agent.policy)
        agent.run_episode(env)
        validate_net(agent.model)
        validate_net(agent.policy)
        

def test_det_qnet():
    net = DeterministicQNetwork()
    x1 = FloatTensor([TradingEnv().reset()])
    for p in range(-1, 2):
        p_tensor = FloatTensor([p])
        qs = net(x1, p_tensor.unsqueeze(0))
    
def test_det_policy():
    net = DeterministicPolicyNetwork()
    x1 = FloatTensor([TradingEnv().reset()])
    pi = net(x1)

def TestDDPG(TestA2C):
    agent_constructor = DDPG
    
    def basic_episode_test(self, agent, **kwargs):
        env = ContinuousTradingEnv(**kwargs)
        validate_net(agent.model)
        validate_net(agent.policy)
        agent.run_episode(env)
        validate_net(agent.model)
        validate_net(agent.policy)
        

@pytest.mark.incremental
class TestModelBased(TestBaseAgent):
    agent_constructor = ModelBasedAgent
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
    
    def basic_episode_test(self, agent, **kwargs):
        env = agent.ENV_CONSTRUCTOR(**kwargs)
        validate_net(agent.R)
        validate_net(agent.T)
        validate_net(agent.Q)
        agent.run_episode(env)
        validate_net(agent.R)
        validate_net(agent.T)
        validate_net(agent.Q)
    
    def test_train_helper(self):
        pass
    
    def test_train_env(self):
        pass


    def test_test_env(self):
        pass
    
        
    def test_three_tickers(self, **kwargs):
        pass

        
@pytest.mark.incremental
class TestModelBased_NoText_Agent(TestBaseAgent):
    agent_constructor = ModelBased_NoText_Agent

    def test_reward_model(self):
        pass

    def test_q_text_model(self):
        pass

    def test_transition_model(self):
        pass
    
    def test_train_helper(self):
        pass
    
    
    def test_train_env(self):
        pass


    def test_test_env(self):
        pass

    def test_three_tickers(self):
        pass
    
    def test_model_without_text(self):
        model_without_text = ModelBased_NoText_Agent()
        assert model_without_text.name == "Model-based without Text"
        

        
    