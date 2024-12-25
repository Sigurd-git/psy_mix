import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

# Create the Blackjack environment
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=100000)


class BlackjackAgent:
    def __init__(
        self, learning_rate=0.01, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1
    ):
        # Initialize Q-table as a defaultdict
        self.q_table = defaultdict(lambda: np.zeros(2))  # 2 actions: 0=stand, 1=hit

        # Learning parameters
        self.lr = learning_rate  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon value
        self.gamma = 0.95  # Discount factor

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        # Q-learning update
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state]) if not done else 0
        new_value = (1 - self.lr) * old_value + self.lr * (
            reward + self.gamma * next_max
        )
        self.q_table[state][action] = new_value

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# Training parameters
n_episodes = 100000
agent = BlackjackAgent()
rewards = []

# Training loop
for episode in trange(n_episodes, desc="Training"):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

    rewards.append(episode_reward)

    # Print progress every 10000 episodes
    if (episode + 1) % 10000 == 0:
        avg_reward = np.mean(rewards[-10000:])
        print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")


# Plot the training results
def plot_training_results(rewards):
    # Calculate rolling average
    window = 1000
    rolling_mean = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 6))
    plt.plot(rolling_mean)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()


plot_training_results(rewards)
