from pettingzoo.classic import connect_four_v3
from sparse_q_table import SparseQTable


def test_sparse_qtable():
    env = connect_four_v3.env(render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        break  # We just need the first observation

    qtable = SparseQTable()

    # # Test getting default value
    initial_value = qtable.get_value(observation, 3)
    print(f"Initial Q-value: {initial_value}")

    # # Test setting and getting a value
    qtable.set_value(observation, 3, 1.0)
    new_value = qtable.get_value(observation, 3)
    print(f"Updated Q-value: {new_value}")

    # # Test table size
    print(f"Table size: {qtable.get_size()}")


if __name__ == "__main__":
    test_sparse_qtable()
