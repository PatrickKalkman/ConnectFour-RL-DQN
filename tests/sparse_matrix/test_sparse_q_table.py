import pytest
from pettingzoo.classic import connect_four_v3
from src.sparse_matrix import SparseQTable


@pytest.fixture
def env():
    """Fixture to create and cleanup the Connect Four environment."""
    env = connect_four_v3.env(render_mode=None)  # No rendering during tests
    env.reset()
    yield env
    env.close()


@pytest.fixture
def initial_observation(env):
    """Fixture to get the first observation from the environment."""
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        return observation


@pytest.fixture
def qtable():
    """Fixture to create a fresh SparseQTable instance."""
    return SparseQTable(default_value=0.0)


class TestSparseQTable:
    def test_initialization(self, qtable):
        """Test if Q-table is properly initialized."""
        assert qtable.default_value == 0.0
        assert qtable.get_size() == 0

    def test_default_values(self, qtable, initial_observation):
        """Test if default values are returned for new states."""
        for action in range(7):
            assert qtable.get_value(initial_observation, action) == 0.0

    def test_set_and_get_value(self, qtable, initial_observation):
        """Test setting and retrieving Q-values."""
        # Set a value
        qtable.set_value(initial_observation, 3, 1.0)

        # Check the set value
        assert qtable.get_value(initial_observation, 3) == 1.0

        # Check other actions still have default value
        assert qtable.get_value(initial_observation, 0) == 0.0

    def test_state_key_consistency(self, qtable, initial_observation):
        """Test if the same state gets the same key."""
        # Set a value
        qtable.set_value(initial_observation, 3, 1.0)

        # Get the value again
        value1 = qtable.get_value(initial_observation, 3)
        value2 = qtable.get_value(initial_observation, 3)

        # Both retrievals should give the same value
        assert value1 == value2 == 1.0

    def test_table_size(self, qtable, initial_observation):
        """Test if table size is correctly tracked."""
        # Initially empty
        assert qtable.get_size() == 0

        # Add one state-action pair
        qtable.set_value(initial_observation, 3, 1.0)
        assert qtable.get_size() == 1

        # Setting another action for same state
        qtable.set_value(initial_observation, 4, 1.0)
        assert qtable.get_size() == 1  # Same state, size shouldn't change

    def test_default_value_storage(self, qtable, initial_observation):
        """Test that default values aren't stored in the table."""
        # Set a value equal to default
        qtable.set_value(initial_observation, 3, 0.0)

        # Table should still be empty
        assert qtable.get_size() == 0

    def test_state_key_format(self, qtable, initial_observation):
        """Test if state key is properly formatted."""
        state_key = qtable._get_state_key(initial_observation)

        # Check if it's a tuple of tuples
        assert isinstance(state_key, tuple)
        assert all(isinstance(row, tuple) for row in state_key)

        # Check dimensions (6x7 board)
        assert len(state_key) == 6
        assert all(len(row) == 7 for row in state_key)

        # Check if values are valid (0, 1, or 2)
        assert all(all(val in {0, 1, 2} for val in row) for row in state_key)


def test_create_environment():
    """Test if we can create and use the environment correctly."""
    env = connect_four_v3.env()
    env.reset()

    # Test if we can get an observation
    observation = None
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        break

    assert observation is not None
    assert "observation" in observation
    assert "action_mask" in observation

    env.close()
