# -*- coding: utf-8 -*-
"""主程序：四组对比+可视化"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import *
from benchmark import run_benchmark
from lp_optimizer import run_lp
from ga_optimizer import run_ga
from dqn_agent import train_dqn

os.makedirs('results', exist_ok=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
time_hour = np.arange(T) * dt

# ========== 1. 运行所有方案 ==========
print("="*60)
print("运行基准方案...")
res_bench = run_benchmark()

print("运行线性规划优化...")
res_lp = run_lp()

print("运行遗传算法优化...")
res_ga = run_ga()

print("训练DQN强化学习...")
agent, reward_log = train_dqn()
res_dqn = agent.test()

all_res = [res_bench, res_lp, res_ga, res_dqn]

# ========== 2. 打印指标对比 ==========
print("\n" + "="*60)
print(f"{'方案':<12} {'成本(元)':<10} {'碳排放(kg)':<12} {'绿电率(%)':<10} {'弃光率(%)':<10} {'利用率(%)':<10}")
print("="*60)
for r in all_res:
    print(f"{r['name']:<12} {r['cost']:<10.2f} {r['carbon']:<12.1f} {r['green_rate']:<10.1f} {r['abandon_rate']:<10.1f} {r['utilization']:<10.1f}")

# ========== 3. 可视化 ==========
fig = plt.figure(figsize=(14, 16), dpi=100)

# 图1：功率平衡对比（DQN）
ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(time_hour, PV_BASE, 'y--', label='光伏总出力', linewidth=2)
ax1.bar(time_hour, res_dqn['pv_use'], width=0.2, label='光伏自用', color='#2ecc71', alpha=0.7)
ax1.bar(time_hour, np.maximum(0, -np.array([agent.action_to_power(agent.policy_net(torch.FloatTensor(s).unsqueeze(0)).argmax().item()) for s in np.zeros((T,6))])), width=0.2, label='储能放电', color='#9b59b6', alpha=0.7)
ax1.bar(time_hour, res_dqn['grid_buy'], bottom=res_dqn['pv_use'], width=0.2, label='电网购电', color='#e74c3c', alpha=0.7)
ax1.plot(time_hour, res_dqn['load'], 'k-', label='充电负荷', linewidth=1.5)
ax1.set_title('DQN优化 - 24小时功率平衡', fontsize=13)
ax1.set_ylabel('功率 (kW)')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 图2：SOC曲线对比
ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(np.arange(T+1)*dt, res_bench['soc']/ESS_CAP*100, 'b-', label='基准方案', linewidth=2)
ax2.plot(np.arange(T+1)*dt, res_lp['soc']/ESS_CAP*100, 'g--', label='线性规划', linewidth=2)
ax2.plot(np.arange(T+1)*dt, res_ga['soc']/ESS_CAP*100, 'r-.', label='遗传算法', linewidth=2)
ax2.plot(np.arange(T+1)*dt, res_dqn['soc']/ESS_CAP*100, 'r-.', label='DQN', linewidth=2)
ax2.set_title('储能SOC曲线对比', fontsize=13)
ax2.set_ylabel('SOC (%)')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# 图3：成本对比柱状图
ax3 = fig.add_subplot(3, 2, 3)
names = [r['name'] for r in all_res]
costs = [r['cost'] for r in all_res]
colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
bars = ax3.bar(names, costs, color=colors, alpha=0.8)
ax3.set_title('日综合成本对比', fontsize=13)
ax3.set_ylabel('成本 (元)')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x()+bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 图4：碳排放对比
ax4 = fig.add_subplot(3, 2, 4)
carbons = [r['carbon'] for r in all_res]
bars = ax4.bar(names, carbons, color=colors, alpha=0.8)
ax4.set_title('日碳排放对比', fontsize=13)
ax4.set_ylabel('碳排放 (kgCO₂)')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x()+bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 图5：绿电率&弃光率
ax5 = fig.add_subplot(3, 2, 5)
x = np.arange(len(names))
green = [r['green_rate'] for r in all_res]
abandon = [r['abandon_rate'] for r in all_res]
ax5.bar(x-0.2, green, 0.4, label='绿电消纳率', color='#2ecc71', alpha=0.8)
ax5.bar(x+0.2, abandon, 0.4, label='弃光率', color='#e67e22', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(names)
ax5.set_title('绿电消纳率与弃光率', fontsize=13)
ax5.set_ylabel('百分比 (%)')
ax5.legend(fontsize=9)

# 图6：DQN训练奖励曲线
ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(reward_log, color='#3498db', linewidth=0.8)
ax6.set_title('DQN训练奖励曲线', fontsize=13)
ax6.set_xlabel('回合数')
ax6.set_ylabel('累计奖励')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/full_comparison.png', bbox_inches='tight')
print("\n✅ 对比图已保存至 results/full_comparison.png")
plt.show()
