#这个文件仍然是非湿润环境，但是使用了REINFORCE
#仅仅是一个学习REINFORCE的练习
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

class MLP(nn.Module):
    def __init__(self, state_dim,hidden_dim,action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, lr,gamma,device):
        self.policy = MLP(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
    
    def take_action(self, state):
        #将状态转换为one-hot编码，最开始的时候忘记了
        state_onehot = torch.zeros(state_dim).to(self.device)
        state_onehot[state] = 1.0
        state = state_onehot.unsqueeze(0)
        action_prob = self.policy(state)
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)
    
    def update(self, rewards, log_probs):
        discounted_rewards = []
        cumulative_reward = 0
        for r in rewards[::-1]:
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * discounted_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

learning_rate = 1e-3 
num_episodes = 3000
hidden_dim = 128 
gamma = 0.98 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env=gym.make('FrozenLake-v1',is_slippery=False)
state_dim=env.observation_space.n
action_dim=env.action_space.n
agent=REINFORCE(state_dim,hidden_dim,action_dim,learning_rate,gamma,device)

success_count = 0

for i in range(num_episodes):
    state, _ = env.reset()
    rewards = []
    log_probs = []
    done = False
    while not done:
        action, log_prob = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state
        done = terminated or truncated
    agent.update(rewards, log_probs)
    if rewards[-1]==1:   
        success_count += 1
    if (i + 1) % 100 == 0:
        win_rate = success_count / (i + 1) * 100
        print(f"Episode {i+1}/{num_episodes} completed. Win rate: {win_rate:.2f}%")

state, _ = env.reset()
done = False
while not done:
    action, _ = agent.take_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")
    state = next_state
    done = terminated or truncated





