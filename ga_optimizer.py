# -*- coding: utf-8 -*-
"""遗传算法优化：启发式算法对比"""

import numpy as np
from config import *
from env import ParkingEnv

def run_ga():
    env = ParkingEnv(random=False)
    load = env.get_daily_load()
    pv = PV_BASE

    # 个体：96维，每个时刻储能功率（-1~1对应-200~200kW）
    def decode(ind):
        return ind * ESS_POWER

    def fitness(ind):
        ess = decode(ind)
        soc = SOC_INIT * ESS_CAP
        cost = 0
        for t in range(T):
            p = ess[t]
            if p > 0:
                grid = max(0, load[t] + p - pv[t])
                soc += p * ETA_CH * dt
            else:
                grid = max(0, load[t] - pv[t] + p)
                soc -= (-p) / ETA_DIS * dt

            if soc < SOC_MIN*ESS_CAP or soc > SOC_MAX*ESS_CAP:
                return 1e6

            cost += grid * PRICE[t] * dt + grid * CARBON_FACTOR * CARBON_PRICE * dt

        # 日循环惩罚
        cost += abs(soc - SOC_INIT*ESS_CAP) * 2
        return cost

    # 初始化种群
    pop = np.random.uniform(-1, 1, (GA_POP, T))

    for _ in range(GA_ITER):
        # 选择
        fit = np.array([fitness(ind) for ind in pop])
        idx = fit.argsort()
        pop = pop[idx]

        # 交叉
        new_pop = pop[:10].tolist()
        while len(new_pop) < GA_POP:
            p1, p2 = pop[np.random.choice(10, 2, replace=False)]
            if np.random.rand() < GA_CROSS:
                point = np.random.randint(0, T)
                child = np.concatenate([p1[:point], p2[point:]])
            else:
                child = p1.copy()
            new_pop.append(child)

        # 变异
        for i in range(10, len(new_pop)):
            if np.random.rand() < GA_MUT:
                point = np.random.randint(0, T)
                new_pop[i][point] = np.random.uniform(-1, 1)

        pop = np.array(new_pop)

    # 最优个体
    best = pop[0]
    ess = decode(best)

    # 计算结果
    soc = np.zeros(T+1)
    soc[0] = SOC_INIT * ESS_CAP
    grid = np.zeros(T)
    pv_use = np.zeros(T)

    for t in range(T):
        p = ess[t]
        if p > 0:
            grid[t] = max(0, load[t] + p - pv[t])
            pv_use[t] = min(pv[t], load[t] + p)
            soc[t+1] = soc[t] + p * ETA_CH * dt
        else:
            grid[t] = max(0, load[t] - pv[t] + p)
            pv_use[t] = min(pv[t], load[t] - p)
            soc[t+1] = soc[t] - (-p) / ETA_DIS * dt

    total_cost = np.sum(grid * PRICE * dt + grid * CARBON_FACTOR * CARBON_PRICE * dt)
    total_carbon = np.sum(grid * CARBON_FACTOR * dt)
    total_load = np.sum(load) * dt
    total_pv = np.sum(pv) * dt
    total_pv_use = np.sum(pv_use) * dt

    return {
        'name': '遗传算法',
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
