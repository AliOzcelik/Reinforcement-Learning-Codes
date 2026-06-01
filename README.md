# Reinforcement Learning Codes

A collection of deep reinforcement learning implementations using various algorithms on different environments.

### Setup

**Windows**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux and macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
Check for CUDA or CPU installation of torch library in the requirements.txt files.

## Contents

### Deep Q-Learning
Implementation of Deep Q-Learning algorithm applied to the Atari `Boxing` environment. Includes trained agent demonstration videos.

### Deep Expected Sarsa
Deep Expected Sarsa algorithm implementation for the Atari `Boxing` environment, providing an alternative approach to value-based RL.

### Deep Convolutional Q-Learning
Convolutional neural network-based Q-Learning implementation for the Atari `Kung Fu Master` environment, demonstrating CNNs for visual input processing.

### TD3 (Twin Delayed Deep Deterministic Policy Gradient)
TD3 algorithm implementations for continuous control tasks across multiple MuJoCo environments:

- `Ant-v5`: Quadruped locomotion task
- `HalfCheetah-v5`: Bipedal robot running task
- `HumanoidStandup-v4`: Humanoid standing-up task

### SAC (Soft Actor-Critic)
SAC algorithm implementation for continuous control on the `Ant-v5` MuJoCo environment. Includes training code, saved model checkpoints, reward plots, and final trained-agent demonstration videos.

## Algorithms Implemented

- **DQN (Deep Q-Network)**: Value-based method for discrete action spaces
- **Deep Expected Sarsa**: On-policy temporal difference learning with deep networks
- **Convolutional DQN**: DQN with CNN architecture for image-based observations
- **TD3**: State-of-the-art actor-critic method for continuous control
- **SAC**: Entropy-regularized actor-critic method for continuous control

## Requirements

- Python 3.x
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib
- tqdm
- Jupyter Notebook

## Usage

Each implementation is self-contained within its respective directory. Navigate to the desired algorithm folder and run the corresponding notebook or Python script.
