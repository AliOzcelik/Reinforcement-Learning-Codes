import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
warnings.simplefilter('ignore')


class ReplayBuffer:

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)
    


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.actor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, state):
        actor_features = self.actor_network(state)
        action_mean = self.actor_mean(actor_features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        value = self.critic_network(state)
        return action_mean, action_std, value
    


class PPO:

    def __init__(self, state_dim, action_dim, device, hidden_dim=256, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 epochs=10, batch_size=64, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, eps=1e-5):

        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.buffer = ReplayBuffer()

        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr, eps=eps)
        
        
    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.store(torch.as_tensor(state, dtype=torch.float32), action, log_prob, reward, value, done)

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_mean, action_std, value = self.policy(state)
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

        action_np = action.cpu().numpy()
        action_clipped = np.clip(action_np, -1.0, 1.0)
        action = action.cpu()
        log_prob = log_prob.cpu() if log_prob is not None else None
        value = value.squeeze(-1).cpu()
        return action_clipped, action, log_prob, value
        #return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, states, actions):
        action_mean, action_std, value = self.policy(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy


    def compute_gae(self, last_value):
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        gae = 0.0

        values_extended = values + [last_value]

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values_extended[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        returns = advantages + values_tensor
        return advantages, returns
    
    def update(self, last_value):
        if len(self.buffer) == 0:
            return

        advantages, returns = self.compute_gae(last_value)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                log_probs, values, entropy = self.evaluate_actions(batch_states, batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)

                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))



def train(env_name="LunarLander-v3", num_steps=1_000_000, steps_per_update=2048, eval_freq=10, video_freq=100, video_folder="videos"):

    env = gym.make(env_name, continuous=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    agent = PPO(state_dim, action_dim, device)

    episode_rewards = []
    episode_reward = 0
    episode_count = 0
    best_avg_reward = -float('inf')

    state, _ = env.reset()
    total_steps = 0

    pbar = tqdm(total=num_steps, desc="Training")

    while total_steps < num_steps:
        for _ in range(steps_per_update):
            action_clipped, action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated

            agent.store(state, action, log_prob, reward, value.item(), done)

            episode_reward += reward
            state = next_state
            total_steps += 1
            pbar.update(1)

            if done:
                episode_rewards.append(episode_reward)
                episode_count += 1

                if episode_count % eval_freq == 0:
                    avg_reward = np.mean(episode_rewards[-eval_freq:])
                    pbar.set_postfix({
                        'ep': episode_count,
                        'avg_reward': f'{avg_reward:.1f}'
                    })

                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        agent.save("best_ppo_clean.pth")

                if episode_count % video_freq == 0:
                    record_video(agent, env_name, video_folder, episode_count)

                episode_reward = 0
                state, _ = env.reset()

            if total_steps >= num_steps:
                break

        _, _, _, last_value = agent.select_action(state)
        agent.update(last_value.item() if not done else 0.0)

    pbar.close()
    env.close()

    return episode_rewards


def record_video(agent, env_name, video_folder, episode_num):
    os.makedirs(video_folder, exist_ok=True)
    video_env = gym.make(env_name, continuous=True, render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder,
                           episode_trigger=lambda x: True,
                           name_prefix=f"ppo_ep{episode_num}")

    state, _ = video_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _, _, _ = agent.select_action(state, deterministic=True)
        state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
        total_reward += reward

    video_env.close()
    print(f"\nVideo saved (ep {episode_num}): reward = {total_reward:.1f}")


def plot_rewards(rewards, window=50):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')

    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, linewidth=2,
                label=f'{window}-Episode Moving Average')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('PPO Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_ppo_clean.png', dpi=150)
    plt.show()

    print(f"\nFirst 10 avg: {np.mean(rewards[:10]):.1f}")
    print(f"Last 10 avg: {np.mean(rewards[-10:]):.1f}")
    print(f"Best: {np.max(rewards):.1f}")


if __name__ == "__main__":
    rewards = train(
        env_name="LunarLander-v3",
        num_steps=1_000_000,
        steps_per_update=2048,
        eval_freq=10,
        video_freq=100,
        video_folder="videos"
    )
    plot_rewards(rewards)