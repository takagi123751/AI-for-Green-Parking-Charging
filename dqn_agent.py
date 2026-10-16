# -*- coding: utf-8 -*-
"""DQN强化学习智能体"""

import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from env import ParkingEnv

class QNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 11
        self.epsilon = EPSILON
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.policy_net = QNet()
        self.target_net = QNet()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

    def action_to_power(self, action):
        # 11个动作：-200 ~ +200kW，步长40kW
        return (action - 5) * 40.0

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s_, done = zip(*batch)

        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r)
        done = torch.FloatTensor(done)

        q_eval = self.policy_net(s).gather(1, a)
        q_next = self.target_net(s_).max(1)[0].detach()
        q_target = r + GAMMA * q_next * (1 - done)

        loss = self.criterion(q_eval.squeeze(), q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def test(self):
        env = ParkingEnv(random=False)
        state = env.reset()
        load = env.get_daily_load()
        pv = PV_BASE

        grid = np.zeros(T)
        pv_use = np.zeros(T)
        soc = np.zeros(T+1)
        soc[0] = SOC_INIT * ESS_CAP

        for t in range(T):
            action = self.act(state)
            power = self.action_to_power(action)
            next_state, reward, done, info = env.step(power)
            grid[t] = info['grid_buy']
            pv_use[t] = info['pv_use']
            soc[t+1] = info['soc']
            state = next_state

        total_cost = np.sum(grid * PRICE * dt + grid * CARBON_FACTOR * CARBON_PRICE * dt)
        total_carbon = np.sum(grid * CARBON_FACTOR * dt)
        total_load = np.sum(load) * dt
        total_pv = np.sum(pv) * dt
        total_pv_use = np.sum(pv_use) * dt

        return {
            'name': 'DQN强化学习',
            'load': load,
            'grid_buy': grid,
            'pv_use': pv_use,
            'soc': soc,
            'cost': total_cost,
            'carbon': total_carbon,
            'green_rate': total_pv_use / total_load * 100,
            'abandon_rate': (total_pv - total_pv_use) / total_pv * 100,
            'utilization': total_load / (MAX_LOAD * 24) * 100
        }

def train_dqn():
    env = ParkingEnv(random=True)
    agent = DQNAgent()
    reward_log = []

    for ep in range(EPISODES):
        state = env.reset()
        total_r = 0
        done = False

        while not done:
            action = agent.act(state)
            power = agent.action_to_power(action)
            next_state, reward, done, info = env.step(power)
            agent.store(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_r += reward

        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        reward_log.append(total_r)
        if ep % 50 == 0:
            print(f"Episode {ep}, Reward: {total_r:.2f}, Epsilon: {agent.epsilon:.3f}")

    return agent, reward_log
