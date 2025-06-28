# ADAPT: Adaptive PID Controller using TD3

ADAPT (Adaptive PID Controller using Twin Delayed Deep Deterministic Policy Gradient) is a reinforcement learning-based approach to automatically tune PID controllers for dynamic systems.

## Overview

This project implements a TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent that learns to control PID parameters in real-time to maintain optimal system performance across varying operating conditions.

## Project Structure

```
ADAPT/
├── Network/                 # Neural network components
│   ├── Actor.py            # Actor network for action selection
│   ├── Critic.py           # Critic networks for value estimation
│   ├── GaussianNoiseGenerator.py  # Noise generation for exploration
│   ├── Logger.py           # Training logging utilities
│   └── ReplayBuffer.py     # Experience replay buffer
├── PIDSim/                 # PID simulation environment
│   └── MultiSystemPIDEnv.py # Multi-system PID environment
├── TD3/                    # TD3 agent implementation
│   ├── TD3Agent.py         # Main TD3 agent class
│   └── clientSide.py       # Training and evaluation client
├── SavedModels/            # Trained model checkpoints
└── train.ipynb            # Jupyter notebook for training
```

## Features

- **TD3 Algorithm**: State-of-the-art actor-critic reinforcement learning
- **Multi-System Support**: Handles multiple PID-controlled systems simultaneously
- **Adaptive Tuning**: Real-time PID parameter adjustment based on system performance
- **Hyperparameter Optimization**: Automated hyperparameter tuning with grid search
- **Model Persistence**: Comprehensive checkpointing and model saving system
- **Performance Visualization**: Training progress and performance analysis tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/a-warraich/ADAPT.git
cd ADAPT
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib gym pandas seaborn
```

## Usage

### Basic Training

Run the training notebook:
```bash
jupyter notebook train.ipynb
```

### Hyperparameter Tuning

The system includes automated hyperparameter tuning:

```python
from TD3.clientSide import run_hyperparameter_tuning

# Run hyperparameter optimization
best_params = run_hyperparameter_tuning(
    learning_rates=[1e-4, 3e-4, 1e-3],
    batch_sizes=[64, 128, 256],
    # ... other parameters
)
```

### Model Evaluation

```python
from TD3.clientSide import evaluate_model

# Evaluate a trained model
performance = evaluate_model('path/to/model/checkpoint')
```

## Configuration

Key parameters can be configured in the training setup:

- **Learning Rates**: Actor and critic learning rates
- **Batch Size**: Training batch size
- **Buffer Size**: Experience replay buffer size
- **Noise Parameters**: Exploration noise settings
- **Update Frequencies**: Target network update intervals

## Model Architecture

- **Actor Network**: 3-layer neural network mapping states to PID parameters
- **Critic Networks**: Twin critic networks for robust value estimation
- **Target Networks**: Delayed target networks for stable training

## Results

The system typically achieves:
- Rapid convergence to optimal PID parameters
- Robust performance across varying system dynamics
- Consistent control performance in multi-system scenarios

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{adapt2024,
  title={ADAPT: Adaptive PID Controller using TD3},
  author={Warraich, A.},
  year={2024},
  url={https://github.com/a-warraich/ADAPT}
}
``` 