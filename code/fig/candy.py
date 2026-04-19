import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Arial'
# 数据
models = ['LightGBM', 'XGBoost', 'ET', 'Bagging_DT',
          'SoftVoting', 'HardVoting', 'Stacking_LGBM', 'Stacking_XGB']
acc = [0.4483, 0.4539, 0.4607, 0.4739, 0.4404, 0.4413, 0.4255, 0.4410]

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制竖线（从 y=0 到 acc 值）
ax.vlines(x=models, ymin=0, ymax=acc, color='skyblue', linewidth=1.0)

# 绘制圆点（棒棒糖的“糖”）
rank = np.argsort(np.argsort(-np.array(acc)))
sizes = 50 + (len(acc) - 1 - rank) * 50   # 乘以系数使大小差异更明显
ax.scatter(models, acc, color='dodgerblue', s=sizes, zorder=3)

# 添加数值标签（可选）
for i, v in enumerate(acc):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 设置标题和轴标签
ax.set_title('Model Performance Comparison (MCC)', fontsize=14)
ax.set_ylabel('MCC', fontsize=12)
ax.set_xlabel('Model', fontsize=12)

# 旋转 x 轴标签，避免重叠
plt.xticks(rotation=45, ha='right')

# 设置 y 轴范围（留出一点空间给标签）
ax.set_ylim(0.20, 0.6)

# 显示网格（可选）
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()