import numpy as np
import matplotlib.pyplot as plt
import os

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 基本参数
# =========================
c = 1.0                   # 翼弦长，仅保留作为参考
u = 1.0                   # 来流速度，仅保留作为参考
dt = 0.000125               # 时间步长

theta0 = np.radians(30)   # 俯仰角幅值
phi = np.radians(90)      # 起伏运动相位

# =========================
# 直接定义单翼运动参数
# =========================
f = 1 / 4.44                 # 拍动频率 Hz，直接给定
h0 = 1                 # 起伏幅值，直接给定
periods = 8              # 计算周期数

omega = 2 * np.pi * f
T = 1 / f
total_time = periods * T

# 时间序列
t = np.arange(0, total_time + dt, dt)

# 根输出目录
base_output_dir = r'E:/Rensq/DC_explore_F/validation/'

# 输出目录命名
dir_name = f'smaple/'
output_dir = os.path.join(base_output_dir, dir_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# =========================
# 单翼俯仰角
# 前 1/4 周期使用修正形式
# =========================
t_switch = T / 4
mask = t <= t_switch

theta_position = np.zeros_like(t)
theta_velocity = np.zeros_like(t)
theta_acceleration = np.zeros_like(t)

# 前 1/4 周期修正段
theta_position[mask] = theta0 / 2 * np.cos(2 * omega * t[mask]) - theta0 / 2
theta_velocity[mask] = -theta0 * omega * np.sin(2 * omega * t[mask])
theta_acceleration[mask] = -2 * theta0 * omega**2 * np.cos(2 * omega * t[mask])

# 后续周期运动段
theta_position[~mask] = -theta0 * np.sin(omega * t[~mask])
theta_velocity[~mask] = -theta0 * omega * np.cos(omega * t[~mask])
theta_acceleration[~mask] = theta0 * omega**2 * np.sin(omega * t[~mask])

# =========================
# 单翼起伏运动
# =========================
h_position = h0 * np.sin(omega * t + phi)
h_velocity = h0 * omega * np.cos(omega * t + phi)
h_acceleration = -h0 * omega**2 * np.sin(omega * t + phi)

# =========================
# 单翼 x 方向运动
# =========================
x_position = np.full_like(t, 0.0)
x_velocity = np.zeros_like(t)
x_acceleration = np.zeros_like(t)

# =========================
# 数据保存
# =========================
# 输出格式：
# t,
# theta, theta_velocity, theta_acceleration,
# x, x_velocity, x_acceleration,
# h, h_velocity, h_acceleration
data = np.column_stack([
    t,
    theta_position, theta_velocity, theta_acceleration,
    x_position, x_velocity, x_acceleration,
    h_position, h_velocity, h_acceleration
])

out_filename = os.path.join(output_dir, "motion_params.dat")
np.savetxt(out_filename, data, delimiter=' ', fmt='%.8f')

print("已完成单翼运动参数生成")
print(f"拍动频率 f = {f:.4f} Hz")
print(f"拍动周期 T = {T:.4f} s")
print(f"起伏幅值 h0 = {h0:.4f}")
print(f"周期数 periods = {periods}")
print(f"总时间 total_time = {total_time:.4f} s")
print(f"运动参数文件已保存到: {out_filename}")

# =========================
# 绘制俯仰角
# =========================
plt.figure(figsize=(12, 6))
plt.plot(t, np.degrees(theta_position), label='单翼俯仰角')
plt.xlabel('时间 (秒)')
plt.ylabel('俯仰角 (度)')
plt.title(f'单翼俯仰角：f={f} Hz, h0={h0}, periods={periods}')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "pitch_angle_single_wing.png"), dpi=300)
plt.show()

# =========================
# 绘制起伏位移
# =========================
plt.figure(figsize=(12, 6))
plt.plot(t, h_position, label='单翼起伏位移')
plt.xlabel('时间 (秒)')
plt.ylabel('起伏位移')
plt.title(f'单翼起伏运动：f={f} Hz, h0={h0}, periods={periods}')
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "heaving_single_wing.png"), dpi=300)
plt.show()