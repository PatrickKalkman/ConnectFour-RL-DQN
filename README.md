# Connect Four AI: From PettingZoo to Kaggle

![Tests](https://github.com/PatrickKalkman/ConnectFour-RL-DQN.git/workflows/Tests/badge.svg)

This repository contains implementations of Connect Four AI agents using both traditional Q-learning with sparse matrix storage and deep Q-learning approaches. The project demonstrates scaling reinforcement learning from simple to complex game environments.

Read the full article (nothing there, currently in progress): [From Zero to Kaggle Hero: Building a Connect Four AI Champion](https://medium.com/@pkalkman)

## Features

### Two Learning Approaches
- Traditional Q-learning with sparse matrix optimization
- Deep Q-learning with PyTorch
- Performance comparison between methods

### Training Environments
- PettingZoo's Connect Four environment for initial development
- Kaggle Connect X environment for competition

### Visualization
- Real-time training progress monitoring
- Win rate and exploration tracking
- Game replay visualization

## Project Structure
```
connect-four-ai/
├── src/
│   ├── sparse_matrix/
│   │   ├── agent.py          # Sparse matrix Q-learning implementation
│   │   ├── train.py          # Training script for sparse matrix approach
│   │   └── utils.py          # Helper functions
│   ├── deep_learning/
│   │   ├── agent.py          # Deep Q-learning implementation
│   │   ├── model.py          # Neural network architecture
│   │   ├── train.py          # Training script for DQN
│   │   └── utils.py          # Helper functions
│   └── common/
│       ├── visualization.py   # Shared visualization tools
│       └── evaluation.py      # Agent evaluation utilities
```

## Requirements
- Python 3.12
- PyTorch 2.1.1
- PettingZoo 1.24.1
- Kaggle Environments
- Additional dependencies in requirements.txt

## Installation

```bash
# Clone repository
git clone https://github.com/PatrickKalkman/ConnectFour-RL-DQN.git
cd ConnectFour-RL-DQN



# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Train Sparse Matrix Agent
```python
from src.sparse_matrix.train import train_sparse_agent

# Train agent
agent = train_sparse_agent(episodes=100000)

# Save trained agent
agent.save("trained_sparse_agent.pkl")
```

### Train Deep Learning Agent
```python
from src.deep_learning.train import train_dqn_agent

# Train agent
agent = train_dqn_agent(episodes=100000)

# Save trained model
agent.save("trained_dqn_model.pth")
```

## Training Progress
- Both agents start with random play
- Sparse matrix approach:
  * Efficiently stores only encountered states
  * Quick learning for common positions
- Deep learning approach:
  * Generalizes to similar board states
  * Better scaling with increased training

## Contributing
Feel free to:
- Submit issues and enhancement requests
- Create pull requests
- Provide feedback on model performance
- Suggest optimization strategies

## License
MIT License - see LICENSE file for details

## Acknowledgments
- PettingZoo team for the Connect Four environment
- Kaggle for the Connect X competition platform
- Connect Four solution theory by James D. Allen and Victor Allis