# -*- coding: utf-8 -*-

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

# 固定 St
st = 0.3

# 不同 k 对应不同 a
CASE_DICT = {
    0.8: 20,
    1.1: 20,
    1.4: 17,
    1.7: 11,
}

# 每个 k 都绘制这四个 dc
# (文件夹中的 dc 标识, 实际 dc 数值)
DC_CASES = [
    (1, 1.0),
    (25, 0.25),
    (50, 0.50),
    (75, 0.75),
]

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow/'
save_root = r'E:/Rensq/DC_explore_F/results_intermittent/transient_results/'
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
    将数值格式化为目录名/文件名使用的字符串
    """
    if float(x).is_integer():
        return str(int(x))
    return f"{x:g}"


def build_force_path(k, st, a, dc_folder, dc_value):
    """
    根据你的目录规则构造 force.dat 路径

    特别注意：
    dc=1 时，最后一级文件夹没有 dc 后缀
    """
    k_str = format_num(k)
    st_str = format_num(st)
    a_str = format_num(a)
    dc_value_str = format_num(dc_value)

    # 前四级目录
    dir_name = os.path.join(
        base_root,
        f'k{k_str}',
        f'st{st_str}k{k_str}',
        f'st{st_str}k{k_str}a{a_str}',
        f'st{st_str}k{k_str}a{a_str}dc{dc_folder}'
    )

    # 最后一级目录名
    if dc_folder == 1:
        # dc=1 时最后一个文件夹没有 dc
        filename = f'k{k_str}st{st_str}a{a_str}'
    else:
        filename = f'k{k_str}st{st_str}a{a_str}dc{dc_value_str}'

    force_dir = os.path.join(dir_name, filename)
    force_path = os.path.join(force_dir, 'force.dat')

    return force_path


def process_one_case(k, st, a, dc_folder, dc_value):
    k_str = format_num(k)
    st_str = format_num(st)
    a_str = format_num(a)
    dc_value_str = format_num(dc_value)

    force_path = build_force_path(k, st, a, dc_folder, dc_value)

    if not os.path.exists(force_path):
        print(f'[跳过] 找不到文件: {force_path}')
        return

    try:
        # ======================
        # 读取 force.dat
        # ======================
        df = pd.read_csv(
            force_path,
            header=None,
            sep=r'\s+',
            engine='python',
            on_bad_lines='skip'
        )

        if df.shape[1] < 28:
            print(f'[跳过] 文件列数不足 28 列: {force_path}')
            return

        # 时间
        t = df.iloc[:, 0].values

        # 前翼受力
        force_x1 = df.iloc[:, 1].values
        force_y1 = df.iloc[:, 2].values
        torque1  = df.iloc[:, 3].values

        # 后翼受力
        force_x2 = df.iloc[:, 4].values
        force_y2 = df.iloc[:, 5].values
        torque2  = df.iloc[:, 6].values

        # 总受力
        force_x_total_raw = df.iloc[:, 7].values
        force_y_total_raw = df.iloc[:, 8].values
        torque_total_raw  = df.iloc[:, 9].values

        # 前翼运动学
        x1       = df.iloc[:, 10].values
        vx1      = df.iloc[:, 11].values
        ax1      = df.iloc[:, 12].values
        y1       = df.iloc[:, 13].values
        vy1      = df.iloc[:, 14].values
        ay1      = df.iloc[:, 15].values
        r1       = df.iloc[:, 16].values
        v_theta1 = df.iloc[:, 17].values
        a_theta1 = df.iloc[:, 18].values

        # 后翼运动学
        x2       = df.iloc[:, 19].values
        vx2      = df.iloc[:, 20].values
        ax2      = df.iloc[:, 21].values
        y2       = df.iloc[:, 22].values
        vy2      = df.iloc[:, 23].values
        ay2      = df.iloc[:, 24].values
        r2       = df.iloc[:, 25].values
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

        # 前 20000 个点不画
        start_idx = 20000 if len(t) > 20000 else 0

        # ======================
        # 绘图
        # ======================
        fig, axes = plt.subplots(3, 1, figsize=(32, 16), sharex=True)

        # --- 升力图 ---
        axes[0].plot(t[start_idx:], cl1[start_idx:], label='Lift Coefficient Wing 1')
        axes[0].plot(t[start_idx:], cl2[start_idx:], label='Lift Coefficient Wing 2')
        axes[0].plot(t[start_idx:], cl_total[start_idx:], label='Total Lift Coefficient', linestyle='--')
        axes[0].set_ylabel(r'$C_L$')
        axes[0].set_title(
            rf'Lift Coefficient Over Time '
            rf'($k={k_str}$, $St={st_str}$, $a={a_str}$, $dc={dc_value_str}$)'
        )
        axes[0].legend()
        axes[0].grid(True)

        # --- 推力图 ---
        axes[1].plot(t[start_idx:], ct1[start_idx:], label='Thrust Coefficient Wing 1')
        axes[1].plot(t[start_idx:], ct2[start_idx:], label='Thrust Coefficient Wing 2')
        axes[1].plot(t[start_idx:], ct_total[start_idx:], label='Total Thrust Coefficient', linestyle='--')
        axes[1].set_ylabel(r'$C_T$')
        axes[1].set_title(
            rf'Thrust Coefficient Over Time '
            rf'($k={k_str}$, $St={st_str}$, $a={a_str}$, $dc={dc_value_str}$)'
        )
        axes[1].legend()
        axes[1].grid(True)

        # --- 功率图 ---
        axes[2].plot(t[start_idx:], cp1[start_idx:], label='Power Coefficient Wing 1')
        axes[2].plot(t[start_idx:], cp2[start_idx:], label='Power Coefficient Wing 2')
        axes[2].plot(t[start_idx:], cp_total[start_idx:], label='Total Power Coefficient', linestyle='--')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel(r'$C_P$')
        axes[2].set_title(
            rf'Power Coefficient Over Time '
            rf'($k={k_str}$, $St={st_str}$, $a={a_str}$, $dc={dc_value_str}$)'
        )
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        fig_name = f'k{k_str}st{st_str}a{a_str}dc{dc_folder}.png'
        fig_path = os.path.join(save_root, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'[完成] k={k_str}, st={st_str}, a={a_str}, dc={dc_value_str} 的图已保存到: {fig_path}')

    except Exception as e:
        print(f'[报错] k={k_str}, st={st_str}, a={a_str}, dc={dc_value_str}, 错误信息: {e}')


# ======================
# 批量处理全部参数组合
# ======================
for k, a in CASE_DICT.items():
    for dc_folder, dc_value in DC_CASES:
        process_one_case(k, st, a, dc_folder, dc_value)

print(f'全部处理完成，结果已保存到: {save_root}')