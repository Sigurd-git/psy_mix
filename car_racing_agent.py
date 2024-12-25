import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from tqdm import trange
import imageio
import os

# Create output directory for GIFs
os.makedirs("training_videos", exist_ok=True)

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Create environment
env = gym.make("CarRacing-v3", render_mode="rgb_array")


# Define the DQN network
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # CNN layers to process 96x96x3 input images
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the size of flattened features
        self.fc_input_dim = self._get_conv_output()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_output(self):
        # Helper function to calculate conv output dimensions
        x = torch.zeros(1, 3, 96, 96)
        x = self.conv(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Define the Agent
class CarRacingAgent:
    def __init__(self, state_dim, action_dim):
        # Define discrete actions mapping to continuous actions
        self.ACTIONS = {
            0: [0.0, 0.0, 0.0],  # No action
            1: [-1.0, 0.0, 0.0],  # Left
            2: [1.0, 0.0, 0.0],  # Right
            3: [0.0, 1.0, 0.0],  # Gas
            4: [0.0, 0.0, 0.8],  # Brake
        }
        self.ACTION_TO_IDX = {tuple(v): k for k, v in self.ACTIONS.items()}

        self.policy_net = DQN(action_dim).to(device)
        self.target_net = DQN(action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

        self.Transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_idx = random.randrange(5)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                action_idx = q_values.max(1)[1].item()

        return np.array(self.ACTIONS[action_idx], dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        action_tuple = tuple(action)
        action_idx = self.ACTION_TO_IDX.get(
            action_tuple, 0
        )  # Default to 0 if not found
        self.memory.append(self.Transition(state, action_idx, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = self.Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        # Compute next Q values
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def preprocess_state(state):
    """Preprocess the state image"""
    return state.transpose((2, 0, 1)) / 255.0


# Training loop
def train(n_episodes=1000):
    agent = CarRacingAgent(state_dim=(3, 96, 96), action_dim=5)
    episode_rewards = []

    for episode in trange(n_episodes, desc="Training"):
        state, _ = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        frames = []  # Store frames for GIF

        for t in range(1000):  # Max steps per episode
            frames.append(env.render())
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state = preprocess_state(next_state)
            episode_reward += reward

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            if done:
                break

            state = next_state

        episode_rewards.append(episode_reward)

        # Save video every 50 episodes
        if (episode + 1) % 50 == 0:
            imageio.mimsave(f"training_videos/episode_{episode+1}.gif", frames, fps=30)

        # Update target network
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(
            f"Episode {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
        )


if __name__ == "__main__":
    train()
