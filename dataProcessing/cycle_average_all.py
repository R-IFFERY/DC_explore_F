import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}"
})

# =========================
# 基本参数
# =========================
u = 1.0
rou = 1.0
c = 1.0
a = 0
dc = 1

# 可自行修改
# k_list =[1.7]
# st_list = [0.3]

# 示例：
k_list = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
st_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

base_root = r'E:/Rensq/DC_explore_F/data_1Tflow/'
# final_save_root = r'E:/Rensq/DC_explore_F/results_intermittent/cycle_averaged_results/'
final_save_root = r'E:/Rensq/DC_explore_F/results_a0/cycle_averaged_results/'
os.makedirs(final_save_root, exist_ok=True)

# =========================
# 遍历所有参数组合
# =========================
for k in k_list:
    for st in st_list:
        dir_name = base_root + f'k{k}/st{st}k{k}/st{st}k{k}a{a}/st{st}k{k}a{a}dc{int(dc)}/'
        filename = f'k{k}st{st}'
        force_dir = dir_name + filename
        force_path = os.path.join(force_dir, 'force.dat')

        # 检查文件是否存在
        if not os.path.exists(force_path):
            print(f'缺少文件，跳过: st={st}, k={k}, a={a}, dc={int(dc*100)}')
            continue

        print(f'正在处理: st={st}, k={k}, a={a}, dc={int(dc*100)}')

        # =========================
        # 读取新 force.dat
        # 28列格式：
        # 1.time
        # 2.force1(1)前翼阻力
        # 3.force1(2)前翼升力
        # 4.mom前翼力矩
        # 5.force2(1)后翼阻力
        # 6.force2(2)后翼升力
        # 7.mom2后翼力矩
        # 8.force(1)总阻力
        # 9.force(2)总升力
        # 10.mom+mom2总力矩
        # 11~19 前翼运动学
        # 20~28 后翼运动学
        # =========================
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

        # =========================
        # 计算升力 / 推力 / 功率系数
        # 保持你原先整体定义方式：取单双翼平均
        # =========================
        cl1 = force_y1
        cl2 = force_y2
        cl_total = force_y_total_raw / 2.0

        ct1 = -force_x1
        ct2 = -force_x2
        ct_total = -force_x_total_raw / 2.0

        cp1 = -force_y1 * vy1 - torque1 * v_theta1
        cp2 = -force_y2 * vy2 - torque2 * v_theta2
        cp_total = (cp1 + cp2) / 2.0

        # =========================
        # 周期参数
        # =========================
        f = k * u / np.pi / c
        T = 1.0 / f

        t_min = t.min()
        t_max = t.max()
        n_periods = int((t_max - t_min) / T)

        # =========================
        # 存储逐周期结果
        # =========================
        results = []

        for period_idx in range(1, n_periods + 1):
            t_start = t_min + (period_idx - 1) * T
            t_end = t_min + period_idx * T

            # 左闭右开，避免边界重复
            mask = (t >= t_start) & (t < t_end)

            if np.sum(mask) == 0:
                continue

            results.append({
                'period': period_idx,
                't_start': t_start,
                't_end': t_end,
                'sample_count': int(np.sum(mask)),

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

        if len(results) == 0:
            print(f'无有效周期数据，跳过: st={st}, k={k}, a={a}, dc={int(dc*100)}')
            continue

        result_df = pd.DataFrame(results)
        result_df = result_df[result_df['period'] > 1]

        # =========================
        # 只保存到统一文件夹
        # =========================
        csv_path_all = os.path.join(final_save_root, f'k={k}st={st}a={a}dc={int(dc*100)}.csv')
        result_df.to_csv(csv_path_all, index=False, encoding='utf-8-sig', float_format='%.8e')

        # =========================
        # 绘图：上下排列（3张）
        # =========================
        period = result_df['period'].values

        plt.figure(figsize=(3.6, 6.2))

        # -------- 第1张图：升力 --------
        plt.subplot(3, 1, 1)
        plt.plot(period, result_df['cl1_mean'], marker='o', label=r'$\overline{C}_{LF}$')
        plt.plot(period, result_df['cl2_mean'], marker='s', label=r'$\overline{C}_{LH}$')
        plt.plot(period, result_df['cl_total_mean'], marker='^', label=r'$\overline{C}_{LS}$')
        plt.xticks([])
        plt.title(f'$k={k}, St={st}, a={a}, dc={int(dc*100)}$')
        plt.legend()
        plt.grid(True)

        # -------- 第2张图：推力 --------
        plt.subplot(3, 1, 2)
        plt.plot(period, result_df['ct1_mean'], marker='o', label=r'$\overline{C}_{TF}$')
        plt.plot(period, result_df['ct2_mean'], marker='s', label=r'$\overline{C}_{TH}$')
        plt.plot(period, result_df['ct_total_mean'], marker='^', label=r'$\overline{C}_{TS}$')
        plt.xticks([])
        plt.legend()
        plt.grid(True)

        # -------- 第3张图：功率 --------
        plt.subplot(3, 1, 3)
        plt.plot(period, result_df['cp1_mean'], marker='o', label=r'$\overline{C}_{PF}$')
        plt.plot(period, result_df['cp2_mean'], marker='s', label=r'$\overline{C}_{PH}$')
        plt.plot(period, result_df['cp_total_mean'], marker='^', label=r'$\overline{C}_{PS}$')
        plt.xlabel(r'$T$')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        fig_path_png_all = os.path.join(final_save_root, f'k={k}St={st}.png')
        plt.savefig(fig_path_png_all, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'已保存: {csv_path_all}')
        print(f'已保存: {fig_path_png_all}')