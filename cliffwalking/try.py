import gymnasium as gym
import numpy as np

class CliffWalkingEnv: 
    def __init__(self, ncol, nrow): 
        self.nrow = nrow        
        self.ncol = ncol 
        self.x = 0 #  记录当前智能体位置的横坐标
        self.y = self.nrow - 1 #  记录当前智能体位置的纵坐标
 
    def step(self, action): #  
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0) 
        #  定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] 
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0])) 
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1])) 
        next_state = self.y * self.ncol + self.x 
        reward = -1 
        done = False 
        if self.y == self.nrow - 1 and self.x > 0: #  下一个位置在悬崖或者目标
            done = True 
            if self.x != self.ncol - 1: 
                reward = -100 
        return next_state, reward, done
    def reset(self): # , 回归初始状态坐标轴原点在左上角
        self.x = 0 
        self.y = self.nrow - 1 
        return self.y * self.ncol + self.x
class QLearning:
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_table=np.zeros([nrow*ncol,n_action])
        self.n_action=n_action
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action
    
    def best_action(self,state):
        Q_max=np.max(self.Q_table[state])
        a=[0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state,i]==Q_max:
                a[i]=1
        return 1
    def update(self,s0,a0,r,s1):
        td_error=r+self.gamma*self.Q_table[s1].max()-self.Q_table[s0,a0]
        self.Q_table[s0,a0]+=self.alpha*td_error

ncol = 12 
nrow = 4 
env = CliffWalkingEnv(ncol, nrow) 
np.random.seed(0)
epsilon=0.1
alpha=0.1
gamma=0.9
agent=QLearning(ncol,nrow,epsilon,alpha,gamma)
num_episodes=500

return_list=[]
for i in range(10):
    for i_episode in range(int(num_episodes/10)):
        episode_return=0
        state=env.reset()