import numpy as np
import torch
import matplotlib.pyplot as plt
from diffsynth_engine.algorithm.noise_scheduler import ScaledLinearScheduler, ExponentialScheduler, KarrasScheduler, BetaScheduler

schedule_function_list = [ScaledLinearScheduler(), ExponentialScheduler(), KarrasScheduler(), BetaScheduler()]

for i, scheduler in enumerate(schedule_function_list, start=1):
    # 对每个schedule函数执行一次采样
    sampled_timesteps = scheduler.schedule(50)[1]
    
    # 初始化一个计数器数组，用于统计各个整数被选中的次数
    distribution_counter = torch.zeros(1000, dtype=torch.int)
    rounded_sampled_timesteps = torch.floor(sampled_timesteps).to(torch.int64)
    
    distribution_counter[rounded_sampled_timesteps] += 1
    # 创建子图
    plt.subplot(len(schedule_function_list), 1, i)
    plt.bar(x_values := np.arange(1000)[::-1], distribution_counter.numpy()[::-1], width=10.0)
    # 设置图形的标签和标题
    plt.xlim(1000, 0)
    plt.xlabel('Timestep')    
    plt.title(f'Distribution of Selected Timesteps by {scheduler.__class__.__name__}')

# 调整布局以防止重叠
plt.tight_layout()

plt.savefig('plots/scheduler.png', format='png', dpi=600)
# 显示图形
plt.show()