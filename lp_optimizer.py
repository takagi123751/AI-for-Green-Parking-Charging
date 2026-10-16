# -*- coding: utf-8 -*-
"""线性规划优化：全局最优理论上限"""

import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value
from config import *
from env import ParkingEnv

def run_lp():
    env = ParkingEnv(random=False)
    load = env.get_daily_load()
    pv = PV_BASE

    prob = LpProblem("LP_Optimal", LpMinimize)

    # 变量
    grid = [LpVariable(f"grid_{t}", 0) for t in range(T)]
    ch = [LpVariable(f"ch_{t}", 0, ESS_POWER) for t in range(T)]
    dis = [LpVariable(f"dis_{t}", 0, ESS_POWER) for t in range(T)]
    pv_use = [LpVariable(f"pv_{t}", 0) for t in range(T)]
    soc = [LpVariable(f"soc_{t}", SOC_MIN*ESS_CAP, SOC_MAX*ESS_CAP) for t in range(T+1)]

    # 目标
    prob += lpSum([grid[t]*PRICE[t]*dt + grid[t]*CARBON_FACTOR*CARBON_PRICE*dt for t in range(T)])

    # 约束
    prob += soc[0] == SOC_INIT * ESS_CAP
    for t in range(T):
        prob += pv_use[t] + dis[t] + grid[t] == load[t] + ch[t]
        prob += pv_use[t] <= pv[t]
        prob += soc[t+1] == soc[t] + ch[t]*ETA_CH*dt - dis[t]/ETA_DIS*dt
    prob += soc[T] == SOC_INIT * ESS_CAP

    prob.solve()

    grid_res = np.array([value(grid[t]) for t in range(T)])
    pv_res = np.array([value(pv_use[t]) for t in range(T)])
    soc_res = np.array([value(soc[t]) for t in range(T+1)])

    total_cost = value(prob.objective)
    total_carbon = np.sum(grid_res * CARBON_FACTOR * dt)
    total_pv_use = np.sum(pv_res) * dt
    total_load = np.sum(load) * dt
    total_pv = np.sum(pv) * dt

    return {
        'name': '线性规划',
        'load': load,
        'grid_buy': grid_res,
        'pv_use': pv_res,
        'soc': soc_res,
        'cost': total_cost,
        'carbon': total_carbon,
        'green_rate': total_pv_use / total_load * 100,
        'abandon_rate': (total_pv - total_pv_use) / total_pv * 100,
        'utilization': total_load / (MAX_LOAD * 24) * 100
    }
