import numpy as np
import pandas as pd
import os

# =========================
# 基本参数
# =========================
u = 1.0
rou = 1.0
c = 1.0
st = 0.7
k = 1.1
a = 0
dc = 1
# 周期
f = k * u / np.pi / c
T = 1.0 / f

# 选定统计的周期编号（从1开始）
selected_cycle = 14

dir_name = f'E:/Rensq/DC_explore_F/data/k{k}/st{st}k{k}/st{st}k{k}a{a}/st{st}k{k}a{a}dc{int(dc)}/'
filename = f'k{k}st{st}'
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

# 绘制前后翼运动学图
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 15))
plt.subplot(3, 2, 1)
plt.plot(t, x1, label='x1 (Wing 1)', color='blue')
plt.plot(t, x2, label='x2 (Wing 2)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.title('Wing x Position')
plt.legend()
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(t, y1, label='y1 (Wing 1)', color='blue')
plt.plot(t, y2, label='y2 (Wing 2)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('y (m)')
plt.title('Wing y Position')
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(3, 2, 3)
plt.plot(t, r1, label='r1 (Wing 1)', color='blue')
plt.plot(t, r2, label='r2 (Wing 2)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('r (degrees)')
plt.title('Wing Angle of Attack')
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(3, 2, 4)
plt.plot(t, v_theta1, label='v_theta1 (Wing 1)', color='blue')
plt.plot(t, v_theta2, label='v_theta2 (Wing 2)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('v_theta (degrees/s)')
plt.title('Wing Angular Velocity')
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(3, 2, 5)
plt.plot(t, a_theta1, label='a_theta1 (Wing 1)', color='blue')
plt.plot(t, a_theta2, label='a_theta2 (Wing 2)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('a_theta (degrees/s^2)')
plt.title('Wing Angular Acceleration')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


