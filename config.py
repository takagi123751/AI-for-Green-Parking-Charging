# -*- coding: utf-8 -*-
"""全局参数配置：15分钟步长，微观车辆模型"""

import numpy as np

# ========== 时间尺度 ==========
T = 96                  # 24小时，15分钟步长
dt = 0.25               # 步长：0.25小时 = 15分钟

# ========== 光伏系统 ==========
PV_CAP = 500            # 装机容量 kW
PV_EFF = 0.85
PV_NOISE = 0.12         # 天气随机波动幅度

# 逐15分钟典型晴天出力曲线（归一化）
PV_CURVE = np.array([
    0,0,0,0, 0,0,0,0,
    0.01,0.03,0.08,0.15, 0.25,0.38,0.52,0.65,
    0.76,0.85,0.91,0.95, 0.97,0.98,0.96,0.92,
    0.86,0.78,0.68,0.56, 0.43,0.30,0.18,0.08,
    0.03,0.01,0,0, 0,0,0,0
])
PV_BASE = PV_CAP * PV_CURVE * PV_EFF

# ========== 储能系统 ==========
ESS_CAP = 1000         # 容量 kWh
ESS_POWER = 200        # 最大充放电功率 kW
ETA_CH = 0.95           # 充电效率
ETA_DIS = 0.95          # 放电效率
SOC_MIN = 0.2
SOC_MAX = 0.9
SOC_INIT = 0.5

# ========== 充电桩与车辆 ==========
CHARGER_NUM = 20
CHARGER_POWER = 7      # 单桩功率 kW
MAX_LOAD = CHARGER_NUM * CHARGER_POWER

# 车辆参数
ARRIVAL_RATE = np.array([  # 每小时到达车辆数（早晚高峰）
    2,1,1,1, 2,5,12,18,
    15,10,8,7, 6,7,9,12,
    16,20,18,14, 10,6,3,2
]).repeat(4) / 4         # 转换为15分钟到达率
BATTERY_CAP = 50        # 平均电池容量 kWh
SOC_ARRIVE_LOW = 0.2    # 到达时最低SOC
SOC_ARRIVE_HIGH = 0.6   # 到达时最高SOC
SOC_TARGET = 0.9        # 目标SOC
STAY_HOURS_MIN = 2      # 最短停留时间
STAY_HOURS_MAX = 8      # 最长停留时间

# ========== 经济与碳参数 ==========
# 分时电价（元/kWh）：谷0-8、平8-10/14-18、峰10-14/18-24
PRICE_HOUR = np.array([0.35]*8 + [0.65]*2 + [0.95]*4 + [0.65]*4 + [0.95]*6)
PRICE = PRICE_HOUR.repeat(4)
CARBON_FACTOR = 0.58    # kgCO2/kWh
CARBON_PRICE = 0.05     # 元/kg

# ========== 算法参数 ==========
# DQN
EPISODES = 800
LR = 0.001
GAMMA = 0.98
EPSILON = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.997
MEMORY_SIZE = 3000
BATCH_SIZE = 64
TARGET_UPDATE = 30

# 遗传算法
GA_POP = 50
GA_ITER = 100
GA_CROSS = 0.7
GA_MUT = 0.1
