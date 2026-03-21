import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from train import qnet, DQNAgent



env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

terminated = False
truncated = False
total_reward = 0

while not terminated and not truncated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()

print(f"本回合总奖励: {total_reward}")
env.close()