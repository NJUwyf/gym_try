import gymnasium as gym
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qnet = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
).to(device)

qnet.load_state_dict(torch.load('dqn_cartpole.pth', map_location=device))
qnet.eval() 

env = gym.make("CartPole-v1", render_mode="human")  

for episode in range(5):
    state, _ = env.reset()
    terminated = truncated = False
    total_reward = 0
    while not (terminated or truncated):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = qnet(state_tensor)
            action = q_values.argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    print(f"Episode {episode+1} total reward: {total_reward}")
env.close()