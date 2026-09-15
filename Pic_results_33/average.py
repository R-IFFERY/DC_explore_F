# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# 绘图设置
# ============================================================

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble":
        r"\usepackage{amsmath} "
        r"\usepackage{newtxtext,newtxmath}"
})


# ============================================================
# 基本参数
# ============================================================

u = 1.0
rou = 1.0
c = 1.0

# St 固定为 0.3
st = 0.3

# 不同 k 对应不同平均俯仰角 a
CASE_DICT = {
    0.8: 20,
    1.1: 20,
    1.4: 17,
    1.7: 11,
}

# ------------------------------------------------------------
# DC 设置
#
# 第一个数字：
#   用于 st0.3k0.8a20dcXX 这一层目录
#
# 第二个数字：
#   实际的 Duty Cycle
# ------------------------------------------------------------

DC_CASES = [
    (1, 1.0),
    (25, 0.25),
    (50, 0.50),
    (75, 0.75),
]


# ============================================================
# 路径
# ============================================================

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow/'

final_save_root = (
    r'E:/Rensq/DC_explore_F/results_intermittent/'
    r'averaged_results/'
)

os.makedirs(final_save_root, exist_ok=True)


# ============================================================
# 数字格式化
# ============================================================

def format_num(x):
    """
    将数字转换为与实际文件夹名称一致的形式。

    例如：
        0.8  -> '0.8'
        1.0  -> '1'
        0.50 -> '0.5'
    """
    if float(x).is_integer():
        return str(int(x))

    return f"{x:g}"


# ============================================================
# 构造 force.dat 路径
# ============================================================

def build_force_path(k, st, a, dc_folder, dc_value):
    """
    根据实际目录结构生成 force.dat 路径。

    DC = 1 时：
    ----------------------------------------------------------
    E:/Rensq/DC_explore_F/data_1Tflow/
        k0.8/
        st0.3k0.8/
        st0.3k0.8a20/
        st0.3k0.8a20dc1/
        k0.8st0.3a20/
        force.dat

    DC = 0.5 时：
    ----------------------------------------------------------
    E:/Rensq/DC_explore_F/data_1Tflow/
        k0.8/
        st0.3k0.8/
        st0.3k0.8a20/
        st0.3k0.8a20dc50/
        k0.8st0.3a20dc0.5/
        force.dat
    """

    k_str = format_num(k)
    st_str = format_num(st)
    a_str = format_num(a)
    dc_value_str = format_num(dc_value)

    # 上一级目录
    dir_name = os.path.join(
        base_root,
        f'k{k_str}',
        f'st{st_str}k{k_str}',
        f'st{st_str}k{k_str}a{a_str}',
        f'st{st_str}k{k_str}a{a_str}dc{dc_folder}'
    )

    # --------------------------------------------------------
    # 最后一级目录
    # DC=1 时不带 dc
    # --------------------------------------------------------

    if dc_folder == 1:

        filename = (
            f'k{k_str}'
            f'st{st_str}'
            f'a{a_str}'
        )

    else:

        filename = (
            f'k{k_str}'
            f'st{st_str}'
            f'a{a_str}'
            f'dc{dc_value_str}'
        )

    force_dir = os.path.join(
        dir_name,
        filename
    )

    force_path = os.path.join(
        force_dir,
        'force.dat'
    )

    return force_path


# ============================================================
# 单个工况处理函数
# ============================================================

def process_one_case(k, st, a, dc_folder, dc_value):

    k_str = format_num(k)
    st_str = format_num(st)
    a_str = format_num(a)
    dc_value_str = format_num(dc_value)

    # ========================================================
    # force.dat 路径
    # ========================================================

    force_path = build_force_path(
        k=k,
        st=st,
        a=a,
        dc_folder=dc_folder,
        dc_value=dc_value
    )

    # ========================================================
    # 检查文件
    # ========================================================

    if not os.path.exists(force_path):

        print(
            f'[缺少文件] '
            f'k={k_str}, '
            f'St={st_str}, '
            f'a={a_str}, '
            f'DC={dc_value_str}'
        )

        print(f'路径：{force_path}')
        print()

        return

    print(
        f'正在处理: '
        f'k={k_str}, '
        f'St={st_str}, '
        f'a={a_str}, '
        f'DC={dc_value_str}'
    )

    print(f'读取: {force_path}')

    # ========================================================
    # 读取 force.dat
    # ========================================================

    df = pd.read_csv(
        force_path,
        header=None,
        sep=r'\s+',
        engine='python',
        on_bad_lines='skip'
    )

    # 检查列数
    if df.shape[1] < 28:

        print(
            f'[跳过] force.dat 列数不足 28 列，'
            f'当前列数 = {df.shape[1]}'
        )

        print()
        return

    # ========================================================
    # 时间
    # ========================================================

    t = df.iloc[:, 0].values


    # ========================================================
    # 前翼受力
    # ========================================================

    force_x1 = df.iloc[:, 1].values
    force_y1 = df.iloc[:, 2].values
    torque1 = df.iloc[:, 3].values


    # ========================================================
    # 后翼受力
    # ========================================================

    force_x2 = df.iloc[:, 4].values
    force_y2 = df.iloc[:, 5].values
    torque2 = df.iloc[:, 6].values


    # ========================================================
    # 总受力
    # ========================================================

    force_x_total_raw = df.iloc[:, 7].values
    force_y_total_raw = df.iloc[:, 8].values
    torque_total_raw = df.iloc[:, 9].values


    # ========================================================
    # 前翼运动学
    # ========================================================

    x1 = df.iloc[:, 10].values
    vx1 = df.iloc[:, 11].values
    ax1 = df.iloc[:, 12].values

    y1 = df.iloc[:, 13].values
    vy1 = df.iloc[:, 14].values
    ay1 = df.iloc[:, 15].values

    r1 = df.iloc[:, 16].values
    v_theta1 = df.iloc[:, 17].values
    a_theta1 = df.iloc[:, 18].values


    # ========================================================
    # 后翼运动学
    # ========================================================

    x2 = df.iloc[:, 19].values
    vx2 = df.iloc[:, 20].values
    ax2 = df.iloc[:, 21].values

    y2 = df.iloc[:, 22].values
    vy2 = df.iloc[:, 23].values
    ay2 = df.iloc[:, 24].values

    r2 = df.iloc[:, 25].values
    v_theta2 = df.iloc[:, 26].values
    a_theta2 = df.iloc[:, 27].values


    # ========================================================
    # 计算升力系数
    # ========================================================

    cl1 = force_y1
    cl2 = force_y2

    # 双翼总升力系数
    cl_total = force_y_total_raw / 2.0


    # ========================================================
    # 计算推力系数
    # ========================================================

    ct1 = -force_x1
    ct2 = -force_x2

    # 双翼总推力系数
    ct_total = -force_x_total_raw / 2.0


    # ========================================================
    # 计算功率系数
    # ========================================================

    cp1 = (
        -force_y1 * vy1
        -torque1 * v_theta1
    )

    cp2 = (
        -force_y2 * vy2
        -torque2 * v_theta2
    )

    cp_total = (
        cp1 + cp2
    ) / 2.0


    # ========================================================
    # 周期参数
    # ========================================================

    f = k * u / (np.pi * c)

    T = 1.0 / f

    t_min = t.min()
    t_max = t.max()

    n_periods = int(
        (t_max - t_min) / T
    )

    print(
        f'频率 f = {f:.6f}, '
        f'周期 T = {T:.6f}, '
        f'完整周期数 = {n_periods}'
    )


    # ========================================================
    # 存储逐周期平均结果
    # ========================================================

    results = []

    for period_idx in range(
        1,
        n_periods + 1
    ):

        t_start = (
            t_min
            + (period_idx - 1) * T
        )

        t_end = (
            t_min
            + period_idx * T
        )

        # 左闭右开，防止边界点重复
        mask = (
            (t >= t_start)
            & (t < t_end)
        )

        sample_count = np.sum(mask)

        if sample_count == 0:
            continue

        results.append({

            'period':
                period_idx,

            't_start':
                t_start,

            't_end':
                t_end,

            'sample_count':
                int(sample_count),

            # ----------------------
            # Lift
            # ----------------------

            'cl1_mean':
                np.mean(cl1[mask]),

            'cl2_mean':
                np.mean(cl2[mask]),

            'cl_total_mean':
                np.mean(cl_total[mask]),

            # ----------------------
            # Thrust
            # ----------------------

            'ct1_mean':
                np.mean(ct1[mask]),

            'ct2_mean':
                np.mean(ct2[mask]),

            'ct_total_mean':
                np.mean(ct_total[mask]),

            # ----------------------
            # Power
            # ----------------------

            'cp1_mean':
                np.mean(cp1[mask]),

            'cp2_mean':
                np.mean(cp2[mask]),

            'cp_total_mean':
                np.mean(cp_total[mask]),
        })


    # ========================================================
    # 检查结果
    # ========================================================

    if len(results) == 0:

        print(
            f'[跳过] 无有效周期数据: '
            f'k={k_str}, '
            f'St={st_str}, '
            f'a={a_str}, '
            f'DC={dc_value_str}'
        )

        print()
        return


    result_df = pd.DataFrame(results)


    # ========================================================
    # 去掉第一个周期
    # ========================================================

    result_df = result_df[
        result_df['period'] > 1
    ].copy()

    if len(result_df) == 0:

        print(
            '[跳过] 删除第一个周期后，'
            '没有剩余有效周期。'
        )

        print()
        return


    # ========================================================
    # 保存逐周期平均 CSV
    # ========================================================

    csv_name = (
        f'k={k_str}'
        f'st={st_str}'
        f'a={a_str}'
        f'dc={dc_folder}.csv'
    )

    csv_path_all = os.path.join(
        final_save_root,
        csv_name
    )

    result_df.to_csv(
        csv_path_all,
        index=False,
        encoding='utf-8-sig',
        float_format='%.8e'
    )


    # ========================================================
    # 绘图
    # ========================================================

    period = result_df[
        'period'
    ].values

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.6, 6.2),
        sharex=True
    )


    # ========================================================
    # 1. 升力
    # ========================================================

    axes[0].plot(
        period,
        result_df['cl1_mean'],
        marker='o',
        label=r'$\overline{C}_{LF}$'
    )

    axes[0].plot(
        period,
        result_df['cl2_mean'],
        marker='s',
        label=r'$\overline{C}_{LH}$'
    )

    axes[0].plot(
        period,
        result_df['cl_total_mean'],
        marker='^',
        label=r'$\overline{C}_{LS}$'
    )

    axes[0].set_title(
        rf'$k={k_str},\ '
        rf'St={st_str},\ '
        rf'\alpha_0={a_str}^\circ,\ '
        rf'DC={dc_value_str}$'
    )

    axes[0].legend()
    axes[0].grid(True)


    # ========================================================
    # 2. 推力
    # ========================================================

    axes[1].plot(
        period,
        result_df['ct1_mean'],
        marker='o',
        label=r'$\overline{C}_{TF}$'
    )

    axes[1].plot(
        period,
        result_df['ct2_mean'],
        marker='s',
        label=r'$\overline{C}_{TH}$'
    )

    axes[1].plot(
        period,
        result_df['ct_total_mean'],
        marker='^',
        label=r'$\overline{C}_{TS}$'
    )

    axes[1].legend()
    axes[1].grid(True)


    # ========================================================
    # 3. 功率
    # ========================================================

    axes[2].plot(
        period,
        result_df['cp1_mean'],
        marker='o',
        label=r'$\overline{C}_{PF}$'
    )

    axes[2].plot(
        period,
        result_df['cp2_mean'],
        marker='s',
        label=r'$\overline{C}_{PH}$'
    )

    axes[2].plot(
        period,
        result_df['cp_total_mean'],
        marker='^',
        label=r'$\overline{C}_{PS}$'
    )

    axes[2].set_xlabel(
        r'Cycle'
    )

    axes[2].legend()
    axes[2].grid(True)


    plt.tight_layout()


    # ========================================================
    # 保存图片
    # ========================================================

    fig_name = (
        f'k={k_str}'
        f'St={st_str}'
        f'a={a_str}'
        f'dc={dc_folder}.png'
    )

    fig_path_png_all = os.path.join(
        final_save_root,
        fig_name
    )

    plt.savefig(
        fig_path_png_all,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig)


    # ========================================================
    # 打印信息
    # ========================================================

    print(f'已保存 CSV: {csv_path_all}')
    print(f'已保存图片: {fig_path_png_all}')
    print()


# ============================================================
# 批量处理全部工况
# ============================================================

for k, a in CASE_DICT.items():

    for dc_folder, dc_value in DC_CASES:

        process_one_case(
            k=k,
            st=st,
            a=a,
            dc_folder=dc_folder,
            dc_value=dc_value
        )


print('============================================================')
print('全部工况处理完成')
print(f'结果保存至: {final_save_root}')
print('============================================================')