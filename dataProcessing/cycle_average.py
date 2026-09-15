import numpy as np
import pandas as pd
import os

# =========================
# 基本参数
# =========================
u = 1.0
rou = 1.0
c = 1.0
st = 0.3
k = 1.7
a = 11
dc = 0.01
# 周期
f = k * u / np.pi / c
T = 1.0 / f

# 选定统计的周期编号（从1开始）
selected_cycle = 1

dir_name = f'E:/Rensq/DC_explore_F/data_1Tflow/k{k}/st{st}k{k}/st{st}k{k}a{a}/st{st}k{k}a{a}dc{int(dc*100)}/'
filename = f'k{k}st{st}a{a}'
force_dir = dir_name + filename
save_dir = os.path.join(dir_name, 'One_period')
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(force_dir, 'force.dat')

# =========================
# 读取 force.dat
# =========================
df = pd.read_csv(
    file_path,
    header=None,
    sep=r'\s+',
    engine='python',
    on_bad_lines='skip'
)

# =========================
# 按 28 列格式读取
# =========================
# 1.time
t = df.iloc[:, 0].values

# 前翼受力
force_x1 = df.iloc[:, 1].values
force_y1 = df.iloc[:, 2].values
torque1 = df.iloc[:, 3].values

# 后翼受力
force_x2 = df.iloc[:, 4].values
force_y2 = df.iloc[:, 5].values
torque2 = df.iloc[:, 6].values

# 总受力
force_x_total_raw = df.iloc[:, 7].values
force_y_total_raw = df.iloc[:, 8].values
torque_total_raw = df.iloc[:, 9].values

# 前翼运动学
x1 = df.iloc[:, 10].values
vx1 = df.iloc[:, 11].values
ax1 = df.iloc[:, 12].values
y1 = df.iloc[:, 13].values
vy1 = df.iloc[:, 14].values
ay1 = df.iloc[:, 15].values
r1 = df.iloc[:, 16].values
v_theta1 = df.iloc[:, 17].values
a_theta1 = df.iloc[:, 18].values

# 后翼运动学
x2 = df.iloc[:, 19].values
vx2 = df.iloc[:, 20].values
ax2 = df.iloc[:, 21].values
y2 = df.iloc[:, 22].values
vy2 = df.iloc[:, 23].values
ay2 = df.iloc[:, 24].values
r2 = df.iloc[:, 25].values
v_theta2 = df.iloc[:, 26].values
a_theta2 = df.iloc[:, 27].values

# =========================
# 系数定义（保持原代码方式）
# =========================
# 升力系数
cl1 = force_y1
cl2 = force_y2
cl_total = force_y_total_raw / 2.0   # 保持原逻辑：整体按单双翼平均

# 推力系数
ct1 = -force_x1
ct2 = -force_x2
ct_total = -force_x_total_raw / 2.0  # 保持原逻辑：整体按单双翼平均

# 功率系数
cp1 = -force_y1 * vy1 - torque1 * v_theta1
cp2 = -force_y2 * vy2 - torque2 * v_theta2
cp_total = (cp1 + cp2) / 2.0         # 保持原逻辑：整体取前后翼平均

# =========================
# 周期划分
# =========================
t_min = t.min()
cycle_id = np.floor((t - t_min) / T).astype(int) + 1
unique_cycles = np.unique(cycle_id)

# =========================
# 计算每个周期平均值
# =========================
cycle_results = []

for cyc in unique_cycles:
    mask = (cycle_id == cyc)

    # 跳过点数太少的不完整周期
    if np.sum(mask) < 10:
        continue

    cycle_results.append({
        'cycle': int(cyc),

        'cl1_mean': np.mean(cl1[mask]),
        'cl2_mean': np.mean(cl2[mask]),
        'cl_total_mean': np.mean(cl_total[mask]),

        'ct1_mean': np.mean(ct1[mask]),
        'ct2_mean': np.mean(ct2[mask]),
        'ct_total_mean': np.mean(ct_total[mask]),

        'cp1_mean': np.mean(cp1[mask]),
        'cp2_mean': np.mean(cp2[mask]),
        'cp_total_mean': np.mean(cp_total[mask]),
    })

cycle_df = pd.DataFrame(cycle_results)

# 保存文件1：全部周期平均结果
all_cycle_save_path = os.path.join(save_dir, 'all_cycle_mean_force_power.csv')
cycle_df.to_csv(all_cycle_save_path, index=False, encoding='utf-8-sig', float_format='%.8e')
print(f'全部周期平均结果已保存至: {all_cycle_save_path}')

# =========================
# 提取选定周期平均值
# =========================
if selected_cycle not in cycle_df['cycle'].values:
    raise ValueError(
        f"selected_cycle={selected_cycle} 不在有效周期范围内，可用周期为：{cycle_df['cycle'].tolist()}"
    )

selected_cycle_df = cycle_df[cycle_df['cycle'] == selected_cycle].copy()

# 保存文件2：选定周期平均结果
selected_cycle_save_path = os.path.join(
    save_dir,
    f'selected_cycle_mean.csv'
)
selected_cycle_df.to_csv(selected_cycle_save_path, index=False, encoding='utf-8-sig', float_format='%.8e')
print(f'选定周期平均结果已保存至: {selected_cycle_save_path}')

# =========================
# 终端输出
# =========================
print("\n========== 全部周期平均结果 ==========")
print(cycle_df.to_string(index=False))

print(f"\n========== 选定周期 {selected_cycle} 平均结果 ==========")
print(selected_cycle_df.to_string(index=False))

import matplotlib.pyplot as plt

# =========================
# 绘图：周期平均结果（统一一张图）
# =========================
line_width = 0.8
alpha_solid = 1.0

fig, axes = plt.subplots(3, 1, figsize=(16.5, 9), dpi=300)

# =========================
# 1. 升力
# =========================
axes[0].plot(cycle_df['cycle'], cycle_df['cl1_mean'],
             color='steelblue', marker='o', markersize=3.5,
             linewidth=line_width, alpha=alpha_solid, label='Forewing')

axes[0].plot(cycle_df['cycle'], cycle_df['cl2_mean'],
             color='crimson', marker='s', markersize=3.5,
             linewidth=line_width, alpha=alpha_solid, label='Hindwing')

axes[0].plot(cycle_df['cycle'], cycle_df['cl_total_mean'],
             color='black', marker='^', markersize=4,
             linewidth=line_width * 1.2, alpha=alpha_solid, label='Total')

axes[0].set_ylabel(r'$\overline{C_L}$', rotation=0, labelpad=20)
axes[0].set_xlabel('Cycle')
axes[0].grid(True, linestyle='--', alpha=0.25)
axes[0].legend(frameon=False)

# =========================
# 2. 推力
# =========================
axes[1].plot(cycle_df['cycle'], cycle_df['ct1_mean'],
             color='steelblue', marker='o', markersize=3.5,
             linewidth=line_width, alpha=alpha_solid, label='Forewing')

axes[1].plot(cycle_df['cycle'], cycle_df['ct2_mean'],
             color='crimson', marker='s', markersize=3.5,
             linewidth=line_width, alpha=alpha_solid, label='Hindwing')

axes[1].plot(cycle_df['cycle'], cycle_df['ct_total_mean'],
             color='black', marker='^', markersize=4,
             linewidth=line_width * 1.2, alpha=alpha_solid, label='Total')

axes[1].set_ylabel(r'$\overline{C_T}$', rotation=0, labelpad=20)
axes[1].set_xlabel('Cycle')
axes[1].grid(True, linestyle='--', alpha=0.25)
axes[1].legend(frameon=False)

# =========================
# 3. 功率（新增）
# =========================
axes[2].plot(cycle_df['cycle'], cycle_df['cp1_mean'],
             color='steelblue', marker='o', markersize=3.5,
             linewidth=line_width, alpha=alpha_solid, label='Forewing')

axes[2].plot(cycle_df['cycle'], cycle_df['cp2_mean'],
             color='crimson', marker='s', markersize=3.5,
             linewidth=line_width, alpha=alpha_solid, label='Hindwing')

axes[2].plot(cycle_df['cycle'], cycle_df['cp_total_mean'],
             color='black', marker='^', markersize=4,
             linewidth=line_width * 1.2, alpha=alpha_solid, label='Total')

axes[2].set_ylabel(r'$\overline{C_P}$', rotation=0, labelpad=20)
axes[2].set_xlabel('Cycle')
axes[2].grid(True, linestyle='--', alpha=0.25)
axes[2].legend(frameon=False)

# =========================
# 保存
# =========================
plt.tight_layout(pad=0.5)

fig_save_path = os.path.join(save_dir, 'cycle_mean_CL_CT_CP.png')
plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')

print(f'周期平均综合图已保存至: {fig_save_path}')

plt.show()