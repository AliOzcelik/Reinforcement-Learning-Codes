import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
from collections import deque
import warnings
import os
warnings.simplefilter('ignore')


class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state), np.array(done, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        return action


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SacAgent:

    def __init__(self, state_dim, action_dim, device, capacity=1e6, hidden_dim=256, actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, batch_size=256, alpha=0.2):

        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.replay_buffer = ReplayBuffer(capacity)

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        # Initialize target networks with current network weights
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Use correct parameters for each optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.criterion = nn.MSELoss()

    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            # For evaluation, use mean action
            with torch.no_grad():
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
        else:
            action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action = self.actor.sample(next_state)
            target_q1_next = self.target_critic_1(next_state, next_state_action)
            target_q2_next = self.target_critic_2(next_state, next_state_action)
            target_q_min = torch.min(target_q1_next, target_q2_next)
            target_q = reward + (1 - done) * self.gamma * target_q_min

        # Update Critic 1
        current_q1 = self.critic_1(state, action)
        critic_1_loss = self.criterion(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Update Critic 2
        current_q2 = self.critic_2(state, action)
        critic_2_loss = self.criterion(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update Actor
        new_action = self.actor.sample(state)
        actor_loss = -self.critic_1(state, new_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def create_video_env(env_name, video_folder="videos", episode_trigger=None):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger,
                     name_prefix="sac-agent")
    return env


def record_agent_video(agent, env_name, video_folder="videos", num_episodes=3):
    print(f"\nRecording {num_episodes} episodes...")
    video_env = create_video_env(env_name, video_folder, episode_trigger=lambda x: True)

    for episode in range(num_episodes):
        state, _ = video_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        print(f"Video Episode {episode+1}: Reward = {episode_reward:.2f}")

    video_env.close()
    print(f"Videos saved to: {video_folder}/")


def train_sac_agent(agent, env, num_episodes=100, warmup_steps=1000, update_freq=1,
                   eval_freq=10, video_freq=50, video_folder="videos"):

    episode_rewards = []
    best_reward = -float('inf')
    total_steps = 0

    # Warmup with random actions
    print(f"Warmup phase: Collecting {warmup_steps} random experiences...")
    state, _ = env.reset()
    for _ in tqdm(range(warmup_steps), desc="Warmup"):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    print(f"\nStarting training for {num_episodes} episodes...")

    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            # Select action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Update agent
            if total_steps % update_freq == 0:
                agent.update()

            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            print(f"\nEpisode {episode+1}/{num_episodes} | Avg Reward (last {eval_freq}): {avg_reward:.2f} | Steps: {steps}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.actor.state_dict(), "best_actor.pth")
                print(f"New best model saved! Avg Reward: {best_reward:.2f}")

        # Record video
        if (episode + 1) % video_freq == 0:
            record_agent_video(agent, env.spec.id, video_folder, num_episodes=1)

    return episode_rewards


def plot_rewards(episode_rewards, window=10):
    """Plot training rewards with moving average"""
    plt.figure(figsize=(12, 6))

    episodes = range(1, len(episode_rewards) + 1)
    plt.plot(episodes, episode_rewards, alpha=0.3, label='Episode Reward')

    # Moving average
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(episode_rewards) + 1), moving_avg,
                linewidth=2, label=f'{window}-Episode Moving Average')

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('SAC Training Progress - Reward vs Episode', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_rewards.png', dpi=150)
    plt.show()

    print(f"\nTraining statistics:")
    print(f"  First 10 episodes avg: {np.mean(episode_rewards[:10]):.2f}")
    print(f"  Last 10 episodes avg: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Best episode: {np.max(episode_rewards):.2f}")
    print(f"  Worst episode: {np.min(episode_rewards):.2f}")





# Environment setup
env_name = "Ant-v5"
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create  agent with better hyperparameters
agent = SacAgent(state_dim, action_dim, device)




# Train the agent
episode_rewards = train_sac_agent(
    agent=agent,
    env=env,
    num_episodes=2000,      # More episodes
    warmup_steps=10000,    # Random exploration first
    update_freq=1,         # Update every step
    eval_freq=10,          # Print stats every 10 episodes
    video_freq=50,         # Record video every 50 episodes
    video_folder="videos"
)


# Plot training progress
plot_rewards(episode_rewards, window=10)