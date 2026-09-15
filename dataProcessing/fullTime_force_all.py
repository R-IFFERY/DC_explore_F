import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 基本参数
# ======================
u = 1.0
rou = 1.0
c = 1.0
a = 0
dc = 1

# 参数组合
# k_list = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
# st_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

k_list = [0.5]
st_list = [0.6]

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow/'

# 统一保存文件夹（仅存图片）
# save_root = r'E:\Rensq\DC_explore_F\results_intermittent\full_time_Forceresults'
save_root = r'E:/Rensq/DC_explore_F/results_a0/fig8'
os.makedirs(save_root, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}"
})


def format_num(x):
    """
    将数值格式化为目录名中使用的形式，避免出现过长浮点数。
    """
    return f"{x:.1f}"


def process_one_case(st, k):
    st_str = format_num(st)
    k_str = format_num(k)

    # 原始数据目录
    dir_name = base_root + f'k{k}/st{st}k{k}/st{st}k{k}a{a}/st{st}k{k}a{a}dc{int(dc)}/'
    filename = f'k{k}st{st}'
    force_dir = dir_name + filename
    force_path = os.path.join(force_dir, 'force.dat')

    if not os.path.exists(force_path):
        print(f'[跳过] 找不到文件: {force_path}')
        return

    try:
        # ======================
        # 读取新 force.dat
        # ======================
        df = pd.read_csv(
            force_path,
            header=None,
            sep=r'\s+',
            engine='python',
            on_bad_lines='skip'
        )

        # 时间
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

        # ======================
        # 计算系数
        # ======================
        cl1 = force_y1
        cl2 = force_y2
        cl_total = force_y_total_raw / 2.0

        ct1 = -force_x1
        ct2 = -force_x2
        ct_total = -force_x_total_raw / 2.0

        cp1 = -force_y1 * vy1 - torque1 * v_theta1
        cp2 = -force_y2 * vy2 - torque2 * v_theta2
        cp_total = (cp1 + cp2) / 2.0

        # 前10个点不画
        start_idx = 20000 if len(t) > 20000 else 0

        # ======================
        # 绘图：上下拼接（3张）
        # ======================
        fig, axes = plt.subplots(3, 1, figsize=(32, 16), sharex=True)

        # --- 升力图 ---
        axes[0].plot(t[start_idx:], cl1[start_idx:], label='Lift Coefficient Wing 1')
        axes[0].plot(t[start_idx:], cl2[start_idx:], label='Lift Coefficient Wing 2')
        axes[0].plot(t[start_idx:], cl_total[start_idx:], label='Total Lift Coefficient', linestyle='--')
        axes[0].set_ylabel('Lift Coefficient (Cl)')
        axes[0].set_title(f'Lift Coefficient Over Time (k={k_str}, st={st_str}, a={a}, dc{int(dc)})')
        axes[0].legend()
        axes[0].grid(True)

        # --- 推力图 ---
        axes[1].plot(t[start_idx:], ct1[start_idx:], label='Thrust Coefficient Wing 1')
        axes[1].plot(t[start_idx:], ct2[start_idx:], label='Thrust Coefficient Wing 2')
        axes[1].plot(t[start_idx:], ct_total[start_idx:], label='Total Thrust Coefficient', linestyle='--')
        axes[1].set_ylabel('Thrust Coefficient (Ct)')
        axes[1].set_title(f'Thrust Coefficient Over Time (k={k_str}, st={st_str}, a={a})')
        axes[1].legend()
        axes[1].grid(True)

        # --- 功率图 ---
        axes[2].plot(t[start_idx:], cp1[start_idx:], label='Power Coefficient Wing 1')
        axes[2].plot(t[start_idx:], cp2[start_idx:], label='Power Coefficient Wing 2')
        axes[2].plot(t[start_idx:], cp_total[start_idx:], label='Total Power Coefficient', linestyle='--')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Power Coefficient (Cp)')
        axes[2].set_title(f'Power Coefficient Over Time (k={k_str}, st={st_str}, a={a}, dc{int(dc)})')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        fig_name = f'k{k_str}st{st_str}a{a}dc{int(dc)}.png'
        fig_path = os.path.join(save_root, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'[完成] st={st_str}, k={k_str}, a={a}, dc={int(dc)} 的受力系数图已保存到: {fig_path}')

    except Exception as e:
        print(f'[报错] st={st_str}, k={k_str}, a={a}, dc={int(dc)}, 错误信息: {e}')


# ======================
# 批量处理全部参数组合
# ======================
for k in k_list:
    for st in st_list:
        process_one_case(st, k)

print(f'全部处理完成，结果已保存到: {save_root}')