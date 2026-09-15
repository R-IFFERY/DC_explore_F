# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 绘图设置
# ============================================================
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble":
        r"\usepackage{amsmath} \usepackage{newtxtext,newtxmath}",
})


# ============================================================
# 基本参数
# ============================================================
u = 1.0
rho = 1.0
c = 1.0

# 连续拍动
dc = 1.0

# 本次统一处理 St = 0.3
st = 0.3

# 不同 k 对应的后翼平均俯仰角 a
case_dict = {
    0.5: [13],
    # 0.8: [19, 20, 21],
    # 1.1: [19, 20, 21],
    # 1.4: [16, 17, 18],
    # 1.7: [10, 11, 12],
}


# ============================================================
# 文件路径
# ============================================================
base_root = r"E:/Rensq/DC_explore_F/data_1Tflow"

# 非零俯仰角结果单独保存，避免与 a=0 的结果混合
final_save_root = (
    r"E:/Rensq/DC_explore_F/results_aneq0/"
    r"cycle_averaged_results_selected"
)

os.makedirs(final_save_root, exist_ok=True)


# ============================================================
# 文件夹名称中的浮点数格式
# 防止出现 0.30000000000000004 等问题
# ============================================================
def format_number(value):
    """
    将数值转换为适合文件夹名称的字符串。

    例如：
    0.8 -> '0.8'
    1.0 -> '1'
    20  -> '20'
    """
    return f"{value:g}"


# ============================================================
# 遍历全部指定工况
# ============================================================
for k, a_list in case_dict.items():

    k_str = format_number(k)
    st_str = format_number(st)

    for a in a_list:

        a_str = format_number(a)
        dc_folder_str = format_number(dc)
        dc_percent = int(round(dc * 100))

        # ----------------------------------------------------
        # 构造非零俯仰角工况的文件路径
        #
        # 示例：
        # E:/Rensq/DC_explore_F/data_1Tflow/
        # k0.8/
        # st0.3k0.8/
        # st0.3k0.8a20/
        # st0.3k0.8a20dc1/
        # k0.8st0.3a20/
        # force.dat
        # ----------------------------------------------------
        force_dir = os.path.join(
            base_root,
            f"k{k_str}",
            f"st{st_str}k{k_str}",
            f"st{st_str}k{k_str}a{a_str}",
            f"st{st_str}k{k_str}a{a_str}dc{dc_folder_str}",
            f"k{k_str}st{st_str}a{a_str}",
        )

        force_path = os.path.join(force_dir, "force.dat")

        # ----------------------------------------------------
        # 检查文件是否存在
        # ----------------------------------------------------
        if not os.path.isfile(force_path):
            print(
                f"缺少文件，跳过："
                f"k={k_str}, St={st_str}, a={a_str}, DC={dc_folder_str}"
            )
            print(f"查找路径：{force_path}")
            print("-" * 80)
            continue

        print(
            f"正在处理："
            f"k={k_str}, St={st_str}, a={a_str}, DC={dc_folder_str}"
        )
        print(f"读取文件：{force_path}")

        # ====================================================
        # 读取 force.dat
        #
        # 28 列格式：
        #  1. time
        #  2. force1(1)  前翼阻力
        #  3. force1(2)  前翼升力
        #  4. mom        前翼力矩
        #  5. force2(1)  后翼阻力
        #  6. force2(2)  后翼升力
        #  7. mom2       后翼力矩
        #  8. force(1)   总阻力
        #  9. force(2)   总升力
        # 10. mom+mom2   总力矩
        # 11~19          前翼运动学
        # 20~28          后翼运动学
        # ====================================================
        try:
            df = pd.read_csv(
                force_path,
                header=None,
                sep=r"\s+",
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            print(f"读取失败，跳过：{exc}")
            print("-" * 80)
            continue

        # 检查列数
        if df.shape[1] < 28:
            print(
                f"数据列数不足，跳过。"
                f"当前列数为 {df.shape[1]}，预期至少为 28。"
            )
            print("-" * 80)
            continue

        # 只保留前 28 列
        df = df.iloc[:, :28].copy()

        # 将数据转换为数值；非数值行转换为 NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        # 删除含有 NaN 的无效数据行
        df = df.dropna().reset_index(drop=True)

        if df.empty:
            print("数据文件中没有有效数值行，跳过。")
            print("-" * 80)
            continue

        # ====================================================
        # 提取时间和受力数据
        # ====================================================
        t = df.iloc[:, 0].to_numpy(dtype=float)

        # 前翼受力
        force_x1 = df.iloc[:, 1].to_numpy(dtype=float)
        force_y1 = df.iloc[:, 2].to_numpy(dtype=float)
        torque1 = df.iloc[:, 3].to_numpy(dtype=float)

        # 后翼受力
        force_x2 = df.iloc[:, 4].to_numpy(dtype=float)
        force_y2 = df.iloc[:, 5].to_numpy(dtype=float)
        torque2 = df.iloc[:, 6].to_numpy(dtype=float)

        # 总受力
        force_x_total_raw = df.iloc[:, 7].to_numpy(dtype=float)
        force_y_total_raw = df.iloc[:, 8].to_numpy(dtype=float)

        # ====================================================
        # 前翼运动学
        # ====================================================
        x1 = df.iloc[:, 10].to_numpy(dtype=float)
        vx1 = df.iloc[:, 11].to_numpy(dtype=float)
        ax1 = df.iloc[:, 12].to_numpy(dtype=float)

        y1 = df.iloc[:, 13].to_numpy(dtype=float)
        vy1 = df.iloc[:, 14].to_numpy(dtype=float)
        ay1 = df.iloc[:, 15].to_numpy(dtype=float)

        r1 = df.iloc[:, 16].to_numpy(dtype=float)
        v_theta1 = df.iloc[:, 17].to_numpy(dtype=float)
        a_theta1 = df.iloc[:, 18].to_numpy(dtype=float)

        # ====================================================
        # 后翼运动学
        # ====================================================
        x2 = df.iloc[:, 19].to_numpy(dtype=float)
        vx2 = df.iloc[:, 20].to_numpy(dtype=float)
        ax2 = df.iloc[:, 21].to_numpy(dtype=float)

        y2 = df.iloc[:, 22].to_numpy(dtype=float)
        vy2 = df.iloc[:, 23].to_numpy(dtype=float)
        ay2 = df.iloc[:, 24].to_numpy(dtype=float)

        r2 = df.iloc[:, 25].to_numpy(dtype=float)
        v_theta2 = df.iloc[:, 26].to_numpy(dtype=float)
        a_theta2 = df.iloc[:, 27].to_numpy(dtype=float)

        # ====================================================
        # 计算升力、推力和功率系数
        #
        # 保持原代码定义：
        # 系统总系数取前后翼总和的一半
        # ====================================================

        # 升力系数
        cl1 = force_y1
        cl2 = force_y2
        cl_total = force_y_total_raw / 2.0

        # 推力系数，推力方向与阻力方向相反
        ct1 = -force_x1
        ct2 = -force_x2
        ct_total = -force_x_total_raw / 2.0

        # 输入功率系数
        cp1 = -force_y1 * vy1 - torque1 * v_theta1
        cp2 = -force_y2 * vy2 - torque2 * v_theta2
        cp_total = (cp1 + cp2) / 2.0

        # ====================================================
        # 计算运动周期
        # ====================================================
        f = k * u / (np.pi * c)
        T = 1.0 / f

        t_min = np.min(t)
        t_max = np.max(t)

        total_time = t_max - t_min
        n_periods = int(np.floor(total_time / T))

        print(f"时间范围：{t_min:.8f} 至 {t_max:.8f}")
        print(f"运动频率：f = {f:.8f}")
        print(f"运动周期：T = {T:.8f}")
        print(f"完整周期数：{n_periods}")

        if n_periods < 1:
            print("数据长度不足一个完整周期，跳过。")
            print("-" * 80)
            continue

        # ====================================================
        # 逐周期计算平均值
        # ====================================================
        results = []

        for period_idx in range(1, n_periods + 1):

            t_start = t_min + (period_idx - 1) * T
            t_end = t_min + period_idx * T

            # 左闭右开，避免相邻周期重复使用边界点
            mask = (t >= t_start) & (t < t_end)

            sample_count = int(np.count_nonzero(mask))

            if sample_count == 0:
                continue

            results.append({
                "period": period_idx,
                "t_start": t_start,
                "t_end": t_end,
                "sample_count": sample_count,

                "cl1_mean": np.mean(cl1[mask]),
                "cl2_mean": np.mean(cl2[mask]),
                "cl_total_mean": np.mean(cl_total[mask]),

                "ct1_mean": np.mean(ct1[mask]),
                "ct2_mean": np.mean(ct2[mask]),
                "ct_total_mean": np.mean(ct_total[mask]),

                "cp1_mean": np.mean(cp1[mask]),
                "cp2_mean": np.mean(cp2[mask]),
                "cp_total_mean": np.mean(cp_total[mask]),
            })

        if not results:
            print("没有得到有效的周期平均结果，跳过。")
            print("-" * 80)
            continue

        result_df = pd.DataFrame(results)

        # 删除第一个周期，与原代码保持一致
        result_df = result_df[result_df["period"] > 1].copy()
        result_df.reset_index(drop=True, inplace=True)

        if result_df.empty:
            print("删除第一个周期后没有剩余数据，跳过。")
            print("-" * 80)
            continue

        # ====================================================
        # 保存逐周期平均结果
        # ====================================================
        output_prefix = (
            f"k={k_str}"
            f"st={st_str}"
            f"a={a_str}"
            f"dc={dc_percent}"
        )

        csv_path = os.path.join(
            final_save_root,
            f"{output_prefix}.csv",
        )

        result_df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            float_format="%.8e",
        )

        # ====================================================
        # 绘制逐周期平均结果
        # ====================================================
        period = result_df["period"].to_numpy()

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(3.8, 6.4),
            sharex=True,
        )

        # ----------------------------------------------------
        # 1. 升力系数
        # ----------------------------------------------------
        axes[0].plot(
            period,
            result_df["cl1_mean"],
            marker="o",
            label=r"$\overline{C}_{L,F}$",
        )
        axes[0].plot(
            period,
            result_df["cl2_mean"],
            marker="s",
            label=r"$\overline{C}_{L,H}$",
        )
        axes[0].plot(
            period,
            result_df["cl_total_mean"],
            marker="^",
            label=r"$\overline{C}_{L,S}$",
        )

        axes[0].set_ylabel(r"$\overline{C}_{L}$")
        axes[0].set_title(
            rf"$k={k_str},\ St={st_str},\ "
            rf"\alpha_0={a_str}^\circ,\ DC={dc_folder_str}$"
        )
        axes[0].legend()
        axes[0].grid(True)

        # ----------------------------------------------------
        # 2. 推力系数
        # ----------------------------------------------------
        axes[1].plot(
            period,
            result_df["ct1_mean"],
            marker="o",
            label=r"$\overline{C}_{T,F}$",
        )
        axes[1].plot(
            period,
            result_df["ct2_mean"],
            marker="s",
            label=r"$\overline{C}_{T,H}$",
        )
        axes[1].plot(
            period,
            result_df["ct_total_mean"],
            marker="^",
            label=r"$\overline{C}_{T,S}$",
        )

        axes[1].set_ylabel(r"$\overline{C}_{T}$")
        axes[1].legend()
        axes[1].grid(True)

        # ----------------------------------------------------
        # 3. 功率系数
        # ----------------------------------------------------
        axes[2].plot(
            period,
            result_df["cp1_mean"],
            marker="o",
            label=r"$\overline{C}_{P,F}$",
        )
        axes[2].plot(
            period,
            result_df["cp2_mean"],
            marker="s",
            label=r"$\overline{C}_{P,H}$",
        )
        axes[2].plot(
            period,
            result_df["cp_total_mean"],
            marker="^",
            label=r"$\overline{C}_{P,S}$",
        )

        axes[2].set_xlabel(r"Cycle number")
        axes[2].set_ylabel(r"$\overline{C}_{P}$")
        axes[2].legend()
        axes[2].grid(True)

        # 坐标刻度朝内
        for ax in axes:
            ax.tick_params(
                axis="both",
                which="both",
                direction="in",
                top=True,
                right=True,
            )

        fig.tight_layout()

        figure_path = os.path.join(
            final_save_root,
            f"{output_prefix}.png",
        )

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"已保存 CSV：{csv_path}")
        print(f"已保存图片：{figure_path}")
        print("-" * 80)


print("全部指定工况处理完成。")