import pytest
from pettingzoo.classic import connect_four_v3
from connectfour.sparse_matrix.q_learning_agent import QLearningAgent


@pytest.fixture
def env():
    """Create and cleanup the Connect Four environment."""
    env = connect_four_v3.env()
    env.reset()
    yield env
    env.close()


@pytest.fixture
def empty_observation(env):
    """Get the initial observation from a fresh environment."""
    for agent in env.agent_iter():
        observation, _, _, _, _ = env.last()
        return observation


@pytest.fixture
def agent():
    """Create a fresh Q-learning agent."""
    return QLearningAgent()


class TestQLearningAgent:
    def test_initialization(self, agent):
        assert agent.epsilon == agent.config.initial_epsilon
        assert isinstance(agent.q_table.q_values, dict)

    def test_choose_action_explore(self, agent, empty_observation):
        agent.epsilon = 1.0
        action = agent.choose_action(empty_observation)
        assert 0 <= action <= 6

    def test_choose_action_exploit(self, agent, empty_observation):
        agent.epsilon = 0.0
        action = agent.choose_action(empty_observation)
        assert 0 <= action <= 6

    def test_learn(self, agent, empty_observation):
        action = 3
        reward = 1.0
        done = False

        initial_q = agent.q_table.get_value(empty_observation, action)
        agent.learn(empty_observation, action, reward, empty_observation, done)
        new_q = agent.q_table.get_value(empty_observation, action)

        assert new_q != initial_q

    def test_epsilon_decay(self, agent):
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon == max(
            agent.config.min_epsilon, initial_epsilon * agent.config.epsilon_decay
        )
