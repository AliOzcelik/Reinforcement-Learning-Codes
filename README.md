# Reinforcement Learning Codes

A collection of deep reinforcement learning implementations using various algorithms on different environments.

## Contents

### Deep Q-Learning
Implementation of Deep Q-Learning algorithm applied to the Atari Boxing environment. Includes trained agent demonstration videos.

- `Deep_Q_Learning/q_learning_boxing_self_2.ipynb` - DQN implementation notebook
- `Deep_Q_Learning/q_learning_agent_video_self.mp4` - Agent performance video

### Deep Expected Sarsa
Deep Expected Sarsa algorithm implementation for the Atari Boxing environment, providing an alternative approach to value-based RL.

- `Deep_Expected_Sarsa/Deep_Expected_Sarsa.ipynb` - Deep Expected Sarsa implementation
- `Deep_Expected_Sarsa/deep_expected_sarsa_boxing.mp4` - Training results video

### Deep Convolutional Q-Learning
Convolutional neural network-based Q-Learning implementation for the Atari Kung Fu Master environment, demonstrating CNNs for visual input processing.

- `Deep_Convolutional_Q_Learning/q_learning_conv_kung_fu_self.ipynb` - Convolutional DQN implementation
- `Deep_Convolutional_Q_Learning/q_learning_conv_agent_video_self.mp4` - Agent gameplay video

### TD3 (Twin Delayed Deep Deterministic Policy Gradient)
TD3 algorithm implementations for continuous control tasks across multiple MuJoCo environments:

- **Ant-v5**: Quadruped locomotion task
  - `TD3/ant/TD3.py` - TD3 implementation
  - Includes trained models, learning curves, and episode videos

- **HalfCheetah-v5**: Bipedal robot running task
  - `TD3/half_cheetah/TD3.py` - TD3 implementation
  - Includes trained models, learning curves, and episode videos

- **HumanoidStandup-v4**: Humanoid standing-up task
  - `TD3/humanoid_standup/TD3.py` - TD3 implementation
  - Includes trained models, learning curves, and episode videos

## Algorithms Implemented

- **DQN (Deep Q-Network)**: Value-based method for discrete action spaces
- **Deep Expected Sarsa**: On-policy temporal difference learning with deep networks
- **Convolutional DQN**: DQN with CNN architecture for image-based observations
- **TD3**: State-of-the-art actor-critic method for continuous control

## Requirements

- Python 3.x
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Jupyter Notebook

## Usage

Each implementation is self-contained within its respective directory. Navigate to the desired algorithm folder and run the corresponding notebook or Python script.
