import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import os
import mujoco


# Twin Delayed Deep Deterministic policy gradient algorithm (TD3)

# Actor: state is input, action is output
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, 300),
            nn.ReLU(True),
            nn.Linear(300, action_dim)
        )
        self.max_action = max_action

    def forward(self, x):
        return torch.tanh(self.main(x)) * self.max_action


# Critic class takes inputs actions and states concatenated together
# Critic returns a simple Q value
# use two critics in the same class
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, 300),
            nn.ReLU(True),
            nn.Linear(300, 1)
        )
        self.twin = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, 300),
            nn.ReLU(True),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.main(x), self.twin(x)

    # For the actor loss in the training process
    def Q1(self, s, a):
        return self.main(torch.cat([s, a], 1))


class ExperienceReplay():

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0

        self.states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.dones       = np.zeros((max_size, 1),          dtype=np.float32)

    def push_memory(self, transition):
        state, next_state, action, reward, done = transition
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.next_states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.dones[ind],
        )


class TD3():

    def __init__(self, state_dim, action_dim, max_action, memory_size, device):
        self.device = device
        self.max_action = max_action

        self.actor        = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optim  = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.experience = ExperienceReplay(state_dim, action_dim, memory_size)

        self.critic        = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        return self.actor(state).detach().cpu().numpy().flatten()

    # gamma is discount factor
    # sigma belongs to normal distribution, for exploration, sigma is policy_noise
    # policy_freq is frequency of the delay: in how many iterations actors and critics will be updated
    def train(self, iterations, batch_size=100, sigma=0.2, noise_clip=0.5,
              gamma=0.99, tau=0.005, policy_freq=2):

        for ite in range(iterations):

            # Sample a batch of transitions (s, s', a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.experience.sample(batch_size)
            state      = torch.as_tensor(batch_states,      dtype=torch.float32, device=self.device)
            next_state = torch.as_tensor(batch_next_states, dtype=torch.float32, device=self.device)
            action     = torch.as_tensor(batch_actions,     dtype=torch.float32, device=self.device)
            reward     = torch.as_tensor(batch_rewards,     dtype=torch.float32, device=self.device)
            done       = torch.as_tensor(batch_dones,       dtype=torch.float32, device=self.device)

            next_action = self.actor_target(next_state)

            # Add Gaussian noise to next action and clamp to valid range
            noise = torch.FloatTensor(next_action.shape).normal_(0, sigma).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Two critic targets take (s', a') and return two Q values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = (reward + gamma * target_q * (1 - done)).detach()

            # Two critics take (s, a) and return two Q values
            current_q1, current_q2 = self.critic(state, action)

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            if ite % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Polyak averaging update for actor and critic targets
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(),  f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth",   weights_only=True))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", weights_only=True))





# Hyperparameters
env_name        = "Ant-v5"
seed            = 0
start_timesteps = int(1e4)  # random exploration phase length
eval_freq       = int(5e3)  # evaluate every N timesteps
max_timesteps   = int(5e5)  # total training timesteps
save_models     = True
expl_noise      = 0.1       # exploration noise std
batch_size      = 100
discount        = 0.99
tau             = 0.005
policy_noise    = 0.2
noise_clip      = 0.5
policy_freq     = 2

file_name = "TD3_%s_%s" % (env_name, seed)
print("---------------------------------------")
print("Settings: %s" % file_name)
print("---------------------------------------")


# Directories
os.makedirs("./results", exist_ok=True)
os.makedirs("./models",  exist_ok=True)
os.makedirs("./videos",  exist_ok=True)


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# Environment & agent setup
env = gym.make(env_name)
env.action_space.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3(state_dim, action_dim, max_action, memory_size=int(1e6), device=device)






# Evaluation helper
def evaluate_policy(agent, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.action_space.seed(seed + 100)
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            avg_reward += reward
    eval_env.close()
    avg_reward /= eval_episodes
    print(f"  [Eval] Avg reward over {eval_episodes} episodes: {avg_reward:.2f}")
    return avg_reward







# Video recording helper
def record_video(agent, env_name, video_dir="./videos", num_episodes=3):
    """Run the trained agent and save episodes as MP4 videos."""
    rec_env = gym.make(env_name, render_mode="rgb_array")
    rec_env = gym.wrappers.RecordVideo(
        rec_env,
        video_folder=video_dir,
        episode_trigger=lambda _: True,
        name_prefix="td3"
    )
    for ep in range(num_episodes):
        state, _ = rec_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(np.array(state))
            state, reward, terminated, truncated, _ = rec_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        print(f"  [Video] Episode {ep + 1}: reward = {ep_reward:.2f}")
    rec_env.close()
    print(f"  Videos saved to '{video_dir}/'")



# Training loop
state, _ = env.reset(seed=seed)
episode_reward    = 0.0
episode_timesteps = 0
episode_num       = 0
evaluations       = []

for t in range(int(max_timesteps)):
    episode_timesteps += 1

    # Action selection: random during warm-up, then policy + exploration noise
    if t < start_timesteps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(np.array(state))
        noise  = np.random.normal(0, max_action * expl_noise, size=action_dim)
        action = (action + noise).clip(-max_action, max_action)

    # Step the environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # done_bool = 0 when the episode ends due to time limit (not a true terminal state)
    done_bool = float(terminated)

    # Store transition in replay buffer
    agent.experience.push_memory((state, next_state, action, reward, done_bool))

    state          = next_state
    episode_reward += reward

    # Train after warm-up phase
    if t >= start_timesteps:
        agent.train(
            iterations  = 1,
            batch_size  = batch_size,
            sigma       = policy_noise,
            noise_clip  = noise_clip,
            gamma       = discount,
            tau         = tau,
            policy_freq = policy_freq
        )

    # Handle episode end
    if done:
        print(f"T: {t + 1:>6} | Episode: {episode_num + 1:>4} | "
              f"Steps: {episode_timesteps:>4} | Reward: {episode_reward:>10.2f}")
        state, _          = env.reset()
        episode_reward    = 0.0
        episode_timesteps = 0
        episode_num      += 1

    # Periodic evaluation and model saving
    if (t + 1) % eval_freq == 0:
        avg_r = evaluate_policy(agent, env_name, seed)
        evaluations.append(avg_r)
        np.save(f"./results/{file_name}", evaluations)
        if save_models:
            agent.save(file_name, "./models")

env.close()



# Plot training curve
plt.figure(figsize=(10, 5))
plt.plot(
    [i * eval_freq for i in range(1, len(evaluations) + 1)],
    evaluations
)
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.title(f"TD3 – {env_name}")
plt.tight_layout()
plt.savefig(f"./results/{file_name}_curve.png")
plt.show()



# Record video of the trained agent
record_video(agent, env_name, video_dir="./videos", num_episodes=3)
