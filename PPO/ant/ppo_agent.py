import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.simplefilter('ignore')




class RolloutBuffer:

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def push(self, state, action, logprob, reward, done):
        self.states.append(torch.as_tensor(state, dtype=torch.float32))
        self.actions.append(action.detach().cpu())
        self.logprobs.append(logprob.detach().cpu())
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def __len__(self):
        return len(self.rewards)


class ActorCritic(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor layers
        self.fc_actor = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic layers
        self.fc_critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor
        action_mean = torch.tanh(self.fc_actor(x))
        action_std = torch.exp(torch.clamp(self.log_std, -20, 2)).expand_as(action_mean)
        action_var = action_std.pow(2)
        cov_mat = torch.diag_embed(action_var)
        
        # Critic
        value = self.fc_critic(x)
        return action_mean, cov_mat, value
    

class PPO:
    
    def __init__(self, state_dim, action_dim, device, hidden_dim=256, learning_rate=3e-4,
                 gamma=0.99, ppo_epochs=10, clip=0.2, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5):

        self.device = device
        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss().to(device)
        
    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_mean, action_var, _ = self.policy_old(state)
            if evaluate:
                action = action_mean
                action_logprob = torch.zeros((), dtype=torch.float32, device=self.device)
            else:
                dist = MultivariateNormal(action_mean, action_var)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
                action_logprob = dist.log_prob(action)

        action = torch.clamp(action, -1.0, 1.0)
        return action.detach().cpu().numpy(), action_logprob.detach()

    def store_transition(self, state, action, logprob, reward, done):
        self.buffer.push(state, torch.as_tensor(action, dtype=torch.float32), logprob, reward, done)

    def update(self):
        if len(self.buffer) == 0:
            return

        rewards = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-7)
        
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()
        
        for _ in range(self.ppo_epochs):
            action_means, action_vars, state_values = self.policy(old_states)
            dists = MultivariateNormal(action_means, action_vars)
            logprobs = dists.log_prob(old_actions)
            dist_entropy = dists.entropy()
            state_values = state_values.squeeze(-1)
            
            ratios = torch.exp(logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.criterion(state_values, rewards)
            loss = actor_loss.mean() + self.value_coef * critic_loss - self.entropy_coef * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)
        
        
def create_video_env(env_name, video_folder="videos", episode_trigger=None):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=episode_trigger,
                     name_prefix="ppo-agent")
    return env


def record_agent_video(agent, env_name, video_folder="videos", num_episodes=3):
    print(f"\nRecording {num_episodes} episodes...")
    video_env = create_video_env(env_name, video_folder, episode_trigger=lambda x: True)

    for episode in range(num_episodes):
        state, _ = video_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        print(f"Video Episode {episode+1}: Reward = {episode_reward:.2f}")

    video_env.close()
    print(f"Videos saved to: {video_folder}/")


def train_ppo_agent(agent, env, num_episodes=100, update_timestep=2048,
                    eval_freq=10, video_freq=50, video_folder="videos"):

    episode_rewards = []
    best_reward = -float('inf')
    total_steps = 0

    print(f"\nStarting training for {num_episodes} episodes...")

    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            action, logprob = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, logprob, reward, done)

            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1

            if total_steps % update_timestep == 0:
                agent.update()

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(episode_rewards[-eval_freq:])
            print(f"\nEpisode {episode+1}/{num_episodes} | Avg Reward (last {eval_freq}): {avg_reward:.2f} | Steps: {steps}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save("best_ppo_policy.pth")
                print(f"New best model saved! Avg Reward: {best_reward:.2f}")

        # Record video
        if (episode + 1) % video_freq == 0:
            record_agent_video(agent, env.spec.id, video_folder, num_episodes=1)

    agent.update()
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
    plt.title('PPO Training Progress - Reward vs Episode', fontsize=14)
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

if __name__ == "__main__":
    env_name = "Ant-v5"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    agent = PPO(state_dim, action_dim, device)

    episode_rewards = train_ppo_agent(
        agent=agent,
        env=env,
        num_episodes=2000,
        update_timestep=2048,
        eval_freq=10,
        video_freq=50,
        video_folder="videos"
    )

    env.close()
    plot_rewards(episode_rewards, window=10)
