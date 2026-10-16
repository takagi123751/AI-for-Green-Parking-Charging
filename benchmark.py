# -*- coding: utf-8 -*-
"""基准方案：无储能优化，车辆来了直接充"""

import numpy as np
from config import *
from env import ParkingEnv

def run_benchmark():
    env = ParkingEnv(random=False)
    load = env.get_daily_load()
    pv = PV_BASE

    grid_buy = np.maximum(load - pv, 0)
    pv_use = np.minimum(pv, load)

    total_cost = np.sum(grid_buy * PRICE * dt + grid_buy * CARBON_FACTOR * CARBON_PRICE * dt)
    total_carbon = np.sum(grid_buy * CARBON_FACTOR * dt)
    total_pv = np.sum(pv) * dt
    total_pv_use = np.sum(pv_use) * dt
    total_load = np.sum(load) * dt
    green_rate = total_pv_use / total_load * 100
    abandon_rate = (total_pv - total_pv_use) / total_pv * 100

    return {
        'name': '基准方案',
        'load': load,
        'grid_buy': grid_buy,
        'pv_use': pv_use,
        'soc': np.ones(T+1) * SOC_INIT * ESS_CAP,
        'cost': total_cost,
        'carbon': total_carbon,
        'green_rate': green_rate,
        'abandon_rate': abandon_rate,
        'utilization': total_load / (MAX_LOAD * 24) * 100
    }
