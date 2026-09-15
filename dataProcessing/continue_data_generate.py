import numpy as np
import matplotlib.pyplot as plt
import os

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# =========================
# 参数
# ========================
a = 13
alpha = np.radians(a)
dc = 1                  # 占空比
p = 0                   # 偏移量（当前未使用，保留）
ks = 9                  # 平滑系数（当前未使用，保留）
c = 1.0                 # 翼弦长
u = 1.0                 # 来流速度
dt = 0.0001             # 时间间隔(秒)
theta0 = np.radians(30) # 俯仰角幅值（转换为弧度）
phi = np.radians(90)    # 相位（转换为弧度）

# 参数组
# k_list = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
# st_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# periods_list = [15, 15, 15, 15, 15, 15, 15]
# periods_list = [9, 12, 12, 16, 16, 16, 18]
k_list = [0.5]
st_list = [0.3]
periods_list = [15]
# 建立 k 与周期数的映射
k_periods_map = dict(zip(k_list, periods_list))

# 根输出目录
base_output_dir = r'E:/Rensq/DC_explore_F/data_1Tflow'

# =========================
# 主循环：遍历全部 k 和 st
# =========================
for k in k_list:
    periods = k_periods_map[k]

    for st in st_list:
        # 目录命名保持你的原有风格
        dir_name = f'st{st}k{k}/'
        dir2_name = f'st{st}k{k}a{a}/'
        filename = f'st{st}k{k}a{a}dc{int(dc)}/'
        output_dir = os.path.join(base_output_dir, f'k{k}', dir_name, dir2_name, filename)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # =========================
        # 参数计算
        # =========================
        f1 = k * u / np.pi / c   # 前翼拍动频率(Hz)
        T1 = 1 / f1
        total_time = periods * T1
        omega1 = 2 * np.pi * f1

        h0 = st * u / (2 * f1)   # 初始起伏

        T2 = T1 * dc
        f2 = 1 / T2
        omega2 = 2 * np.pi * f2

        t = np.arange(0, total_time + dt, dt)

        # =========================
        # 前翼俯仰角：前1/4周期使用修正形式
        # =========================
        t_switch1 = T1 / 4
        mask1 = t <= t_switch1

        theta_position1 = np.zeros_like(t)
        theta_velocity1 = np.zeros_like(t)
        theta_acceleration1 = np.zeros_like(t)

        # 前1/4周期
        theta_position1[mask1] = theta0 / 2 * np.cos(2 * omega1 * t[mask1]) - theta0 / 2
        theta_velocity1[mask1] = -theta0 * omega1 * np.sin(2 * omega1 * t[mask1])
        theta_acceleration1[mask1] = -2 * theta0 * omega1**2 * np.cos(2 * omega1 * t[mask1])

        # 后续恢复原公式
        theta_position1[~mask1] = -theta0 * np.sin(omega1 * t[~mask1])
        theta_velocity1[~mask1] = -theta0 * omega1 * np.cos(omega1 * t[~mask1])
        theta_acceleration1[~mask1] = theta0 * omega1**2 * np.sin(omega1 * t[~mask1])

        # =========================
        # 前翼起伏(h1)
        # =========================
        h_position1 = h0 * np.sin(omega1 * t + phi)
        h_velocity1 = h0 * omega1 * np.cos(omega1 * t + phi)
        h_acceleration1 = -h0 * omega1**2 * np.sin(omega1 * t + phi)

        # =========================
        # 前翼x方向
        # =========================
        # x 位置定为-10
        x_position1 = np.full_like(t, -0.0)
        x_velocity1 = np.zeros_like(t)
        x_acceleration1 = np.zeros_like(t)

        # =========================
        # 后翼俯仰角：前1/4周期使用修正形式
        # =========================
        t_switch2 = T2 / 4
        mask2 = t <= t_switch2

        theta_position2 = np.zeros_like(t)
        theta_velocity2 = np.zeros_like(t)
        theta_acceleration2 = np.zeros_like(t)

        # 前1/4周期
        theta_position2[mask2] = theta0 / 2 * np.cos(2 * omega2 * t[mask2]) - theta0 / 2 + alpha
        theta_velocity2[mask2] = -theta0 * omega2 * np.sin(2 * omega2 * t[mask2])
        theta_acceleration2[mask2] = -2 * theta0 * omega2**2 * np.cos(2 * omega2 * t[mask2])

        # 后续恢复原公式
        theta_position2[~mask2] = -theta0 * np.sin(omega2 * t[~mask2]) + alpha
        theta_velocity2[~mask2] = -theta0 * omega2 * np.cos(omega2 * t[~mask2])
        theta_acceleration2[~mask2] = theta0 * omega2**2 * np.sin(omega2 * t[~mask2])

        # =========================
        # 后翼起伏(h2)
        # =========================
        h_position2 = h0 * np.sin(omega2 * t + phi)
        h_velocity2 = h0 * omega2 * np.cos(omega2 * t + phi)
        h_acceleration2 = -h0 * omega2**2 * np.sin(omega2 * t + phi)

        # =========================
        # 后翼x方向
        # =========================
        x_position2 = np.full_like(t, 1.50)
        x_velocity2 = np.zeros_like(t)
        x_acceleration2 = np.zeros_like(t)

        # =========================
        # 数据保存
        # =========================
        data = np.column_stack([
            t,
            theta_position1, theta_velocity1, theta_acceleration1,
            x_position1, x_velocity1, x_acceleration1,
            h_position1, h_velocity1, h_acceleration1,
            theta_position2, theta_velocity2, theta_acceleration2,
            x_position2, x_velocity2, x_acceleration2,
            h_position2, h_velocity2, h_acceleration2
        ])

        out_filename = os.path.join(output_dir, f'k{k}st{st}a{a}.dat')
        # out_filename = os.path.join('E:/Rensq/DC_explore_F/data/all/', f'k{k}st{st}a{a}.dat')
        np.savetxt(out_filename, data, delimiter=' ', fmt='%.8f')

        print(f"已完成: k={k}, st={st}, periods={periods}")
        print(f"前翼拍动频率 f1 = {f1:.4f} Hz, T1 = {T1:.4f} 秒")
        print(f"初始起伏 h0 = {h0:.4f} 米")
        print(f"运动参数文件已保存到: {out_filename}")

print("全部参数组合的数据与图像已生成完成。")

#
# # 绘制前翼和后翼的俯仰角随时间变化的图像
# plt.figure(figsize=(12, 6))
# plt.plot(t, np.degrees(theta_position1), label='前翼俯仰角 (degrees)')
# plt.plot(t, np.degrees(theta_position2), label='后翼俯仰角 (degrees)')
# plt.xlabel('时间 (秒)')
# plt.ylabel('俯仰角 (度)')
# plt.title(f'k={k}, st={st}, periods={periods}')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# # plt.savefig(os.path.join(output_dir, "pitch_angles.png"), dpi=300)
# plt.show()