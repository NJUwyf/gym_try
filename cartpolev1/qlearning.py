import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import gymnasium as gym
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        state,action,reward,next_state,done = zip(*random.sample(self.buffer,batch_size))
        return np.array(state),action,reward,np.array(next_state),done
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update_freq,device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.device = device
        self.qnet = self.build_model().to(self.device)
        self.target_net = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.update_count = 0
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        return model
    def take_action(self,state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action
    def update(self,transition):
        states,actions,rewards,next_states,dones = transition
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        q_values = self.qnet(states).gather(1,actions)
        next_q_values = self.target_net(next_states).max(dim=1,keepdim=True)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values,target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        self.update_count += 1

lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.98
epsilon=0.01
target_update_freq=10
buffer_size=10000
minimal_size=500
batch_size=64

np.random.seed(0)
random.seed(0)
env = gym.make('CartPole-v1')
#env.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_buffer = ReplayBuffer(buffer_size)
dqn = DQN(state_dim=env.observation_space.shape[0],hidden_dim=hidden_dim,action_dim=env.action_space.n,lr=lr,gamma=gamma,epsilon=epsilon,target_update_freq=target_update_freq,device=device)
return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/10),desc='Iteration %d'%i) as pbar:
        for episode in range(int(num_episodes/10)):
            episode_return = 0
            state=env.reset()[0]
            done = False
            while not done:
                action = dqn.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if len(replay_buffer) > minimal_size:
                    transition = replay_buffer.sample(batch_size)
                    dqn.update(transition)
            return_list.append(episode_return)
            if (episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': episode + 1, 'return': np.mean(return_list[-10:])})
            pbar.update(1)

torch.save(dqn.qnet.state_dict(), 'dqn_cartpole.pth')