import gymnasium as gym
import numpy as np
import random

class QLearningAgent:
    def __init__(self, q_table, action_size, lr, gamma, epsilon, epsilon_decay, epsilon_min):
        self.q_table = q_table
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def take_action(self, state, test=False):
        if test or random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return random.randint(0, self.action_size - 1)

    def learn(self, state, action, reward, next_state, done):
        predict = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 环境
train_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=None)
q_table = np.zeros((train_env.observation_space.n, train_env.action_space.n))
action_size = train_env.action_space.n

# 参数
lr = 0.5
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
epochs = 10000

agent = QLearningAgent(q_table, action_size, lr, gamma, epsilon, epsilon_decay, epsilon_min)

# 训练
success_episodes = []
for episode in range(epochs):
    state, info = train_env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, info = train_env.step(action)
        agent.learn(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
        total_reward += reward
    agent.decay_epsilon()
    
    if total_reward > 0:
        success_episodes.append(episode)
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode+1}: success rate = {len(success_episodes) / (episode+1):.2f}, epsilon={agent.epsilon:.4f}")

train_env.close()

# 测试
test_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode=None)
state, info = test_env.reset()
done = False
total_reward = 0
steps = []
while not done:
    action = agent.take_action(state, test=True)
    steps.append(action)
    next_state, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    state = next_state
    done = terminated or truncated
test_env.close()
print(f"Test Total Reward: {total_reward}")
print(f"Test steps: {steps}")