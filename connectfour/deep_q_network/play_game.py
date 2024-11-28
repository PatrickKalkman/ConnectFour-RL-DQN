import os
import time

import torch
from pettingzoo.classic import connect_four_v3

from connectfour.deep_q_network.dqn_agent import DQNAgent


def play_against_agent(model_path: str = None):
    if model_path is None:
        potential_paths = [
            "./models/dqn_agent_self_play.pth",
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

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize DQN agent
    state_dim = (3, 6, 7)  # Channels, Height, Width
    action_dim = 7  # Connect Four has 7 possible actions

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)

    try:
        # Load the model with correct device mapping
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        print("Successfully loaded agent model")
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return

    # Create environment with human render mode
    env = connect_four_v3.env(render_mode="human")
    env.reset()

    def preprocess_observation(obs):
        """Convert observation to DQN format"""
        board = torch.from_numpy(obs["observation"][:, :, 0]).to(device)
        player_pieces = (board == 1).float()
        opponent_pieces = (board == -1).float()
        valid_moves = torch.zeros((6, 7), device=device)
        valid_moves[0, [i for i, valid in enumerate(obs["action_mask"]) if valid]] = 1
        return torch.stack([player_pieces, opponent_pieces, valid_moves])

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if reward != 0:
                winner = "Agent" if agent_name == "player_0" else "Human"
                print(f"\n{winner} wins!")
            else:
                print("\nIt's a draw!")
            break

        if agent_name == "player_1":  # Human player
            # Get valid actions
            valid_actions = [
                i for i, mask in enumerate(observation["action_mask"]) if mask == 1
            ]

            # Show agent's analysis of the current state
            print("\nAgent's evaluation of possible moves:")
            state = preprocess_observation(observation)
            with torch.no_grad():
                q_values = agent.policy_net(state.unsqueeze(0))[0]
                for action in valid_actions:
                    value = q_values[action].item()
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

        else:  # AI agent
            print("\nAgent is thinking...")
            time.sleep(0.5)  # Add a small delay for better visualization

            state = preprocess_observation(observation)
            valid_moves = [
                i for i, valid in enumerate(observation["action_mask"]) if valid
            ]
            action = agent.select_action(state, valid_moves)

            # Show agent's chosen move and its value
            with torch.no_grad():
                q_values = agent.policy_net(state.unsqueeze(0))[0]
                value = q_values[action].item()
                print(f"Agent chose column {action} (value: {value:.3f})")

        env.step(action)

    env.close()

    # Ask if want to play again
    play_again = input("\nWould you like to play again? (y/n): ")
    if play_again.lower() == "y":
        play_against_agent(model_path)


if __name__ == "__main__":
    print("Welcome to Connect Four!")
    print("Agent is player 1 (red), You are player 2 (yellow)")
    play_against_agent()
