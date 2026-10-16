# -*- coding: utf-8 -*-
"""光储充微观停车场环境：车辆泊松到达，真实充电行为"""

import numpy as np
from config import *

class Vehicle:
    def __init__(self, vid, arrive_time):
        self.vid = vid
        self.arrive = arrive_time
        self.soc = np.random.uniform(SOC_ARRIVE_LOW, SOC_ARRIVE_HIGH)
        self.cap = BATTERY_CAP * np.random.uniform(0.8, 1.2)
        self.demand = (SOC_TARGET - self.soc) * self.cap
        self.stay = np.random.uniform(STAY_HOURS_MIN, STAY_HOURS_MAX)
        self.deadline = min(int(arrive_time + self.stay / dt), T)
        self.charged = 0.0
        self.finished = False

class ParkingEnv:
    def __init__(self, random=True):
        self.random = random
        self.reset()

    def reset(self):
        self.t = 0
        self.soc = SOC_INIT * ESS_CAP
        self.vehicles = []
        self.queue = []
        self.vid_counter = 0

        # 当日光伏曲线
        if self.random:
            self.pv = PV_BASE * (1 + np.random.uniform(-PV_NOISE, PV_NOISE, T))
        else:
            self.pv = PV_BASE.copy()
        self.pv = np.clip(self.pv, 0, None)

        # 预生成当日车辆到达序列
        self.arrival_schedule = []
        for t in range(T):
            n = np.random.poisson(ARRIVAL_RATE[t])
            for _ in range(n):
                self.arrival_schedule.append(t)

        return self._get_state()

    def _get_state(self):
        active_num = len([v for v in self.vehicles if not v.finished])
        return np.array([
            self.soc / ESS_CAP,
            self.pv[self.t] / PV_CAP,
            active_num / CHARGER_NUM,
            PRICE[self.t] / PRICE.max(),
            len(self.queue) / 10,
            self.t / T
        ], dtype=np.float32)

    def _update_vehicles(self, charge_power):
        """分配充电功率给车辆，返回实际充电负荷"""
        active = [v for v in self.vehicles if not v.finished]
        if not active:
            return 0.0

        per_power = min(CHARGER_POWER, charge_power / len(active))
        total_charge = 0.0

        for v in active:
            need = v.demand - v.charged
            add = min(per_power * dt, need)
            v.charged += add
            total_charge += add / dt
            if v.charged >= v.demand - 0.01:
                v.finished = True

        # 处理到期车辆
        for v in active:
            if self.t >= v.deadline and not v.finished:
                v.finished = True

        return total_charge

    def step(self, ess_power):
        """
        ess_power：储能功率，负=放电，正=充电，单位kW
        返回：下一状态，奖励，结束，信息
        """
        # 新到达车辆加入
        while self.arrival_schedule and self.arrival_schedule[0] == self.t:
            self.arrival_schedule.pop(0)
            self.vid_counter += 1
            self.vehicles.append(Vehicle(self.vid_counter, self.t))

        # 计算可用功率
        pv_available = self.pv[self.t]
        if ess_power > 0:
            load_need = self._update_vehicles(pv_available + ess_power)
            grid_buy = max(0, load_need + ess_power - pv_available)
            pv_use = min(pv_available, load_need + ess_power)
        else:
            dis_power = -ess_power
            load_need = self._update_vehicles(pv_available + dis_power)
            grid_buy = max(0, load_need - pv_available - dis_power)
            pv_use = min(pv_available, load_need)

        # 更新SOC
        if ess_power > 0:
            self.soc += ess_power * ETA_CH * dt
        else:
            self.soc -= (-ess_power) / ETA_DIS * dt

        # 越界惩罚
        reward = 0
        if self.soc < SOC_MIN * ESS_CAP or self.soc > SOC_MAX * ESS_CAP:
            reward -= 200
            self.soc = np.clip(self.soc, SOC_MIN*ESS_CAP, SOC_MAX*ESS_CAP)

        # 成本奖励
        cost = grid_buy * PRICE[self.t] * dt + grid_buy * CARBON_FACTOR * CARBON_PRICE * dt
        reward -= cost

        # 未满足充电惩罚
        unfinished = len([v for v in self.vehicles if self.t >= v.deadline and not v.finished])
        reward -= unfinished * 50

        self.t += 1
        done = self.t >= T
        next_state = self._get_state() if not done else np.zeros(6)

        info = {
            'grid_buy': grid_buy,
            'pv_use': pv_use,
            'ess_power': ess_power,
            'load': load_need,
            'soc': self.soc,
            'pv_total': self.pv[self.t]
        }

        return next_state, reward, done, info

    def get_daily_load(self):
        """获取当日基准负荷曲线（无储能优化）"""
        self.reset()
        load = np.zeros(T)
        for t in range(T):
            while self.arrival_schedule and self.arrival_schedule[0] == t:
                self.arrival_schedule.pop(0)
                self.vid_counter += 1
                self.vehicles.append(Vehicle(self.vid_counter, t))
            load[t] = self._update_vehicles(MAX_LOAD)
        return load
