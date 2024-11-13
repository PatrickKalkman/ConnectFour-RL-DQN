import os
import time

from pettingzoo.classic import connect_four_v3

from connectfour.sparse_matrix.q_learning_agent import QLearningAgent


def play_against_agent(model_path: str = None):
    if model_path is None:
        # Try different potential model paths
        potential_paths = [
            "./models/agent.npy",
        ]

        for path in potential_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {path}")
                break

        if model_path is None:
            print(
                "Error: Could not find model file. Available files in current directory:"
            )
            print(os.listdir("."))
            model_path = input("Please enter the path to your model file: ")

    # Load the trained agent
    agent = QLearningAgent()
    agent.epsilon = 0  # Set epsilon to 0 to make agent play deterministically

    try:
        agent.load(model_path)
        print(
            f"Successfully loaded agent with Q-table size: {agent.q_table.get_size()}"
        )
        if agent.q_table.get_size() == 0:
            print("Warning: Q-table is empty! The agent won't play optimally.")
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return

    # Create environment with human render mode
    env = connect_four_v3.env(render_mode="human")
    env.reset()

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if reward > 0:
                winner = "Agent" if agent_name == "player_0" else "Human"
                print(f"\n{winner} wins!")
            elif reward < 0:
                winner = "Human" if agent_name == "player_0" else "Agent"
                print(f"\n{winner} wins!")
            else:
                print("\nIt's a draw!")
            break

        if agent_name == "player_1":  # Human player (now player_1)
            # Get valid actions
            valid_actions = [
                i for i, mask in enumerate(observation["action_mask"]) if mask == 1
            ]

            # Show agent's analysis of the current state
            print("\nAgent's evaluation of possible moves:")
            for action in valid_actions:
                value = agent.q_table.get_value(observation, action)
                print(f"Column {action}: {value:.3f}")

            # Get human input
            while True:
                try:
                    print(f"\nValid moves: {valid_actions}")
                    action = int(input("Enter your move (0-6): "))
                    if action in valid_actions:
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter a number between 0 and 6.")
        else:
            print("\nAgent is thinking...")
            time.sleep(0.5)
            action = agent.choose_action(observation)

            # Show agent's chosen move and its value
            value = agent.q_table.get_value(observation, action)
            print(f"Agent chose column {action} (value: {value:.3f})")

        env.step(action)

    env.close()

    play_again = input("\nWould you like to play again? (y/n): ")
    if play_again.lower() == "y":
        play_against_agent(model_path)


if __name__ == "__main__":
    print("Welcome to Connect Four!")
    print(
        "Agent is player 1 (red), You are player 2 (yellow)"
    )  # Updated player information
    play_against_agent()
