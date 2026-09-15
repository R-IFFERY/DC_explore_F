# -*- coding: utf-8 -*-
"""
plot_fig_pod_dual_x_metrics_2x2.py

绘制 2 x 2 排列的 POD 指标图。

橙色上横坐标：
    St = 0.3，k = 0.8、1.1、1.4、1.7；
    k = 0.8、1.1 使用第15周期；
    k = 1.4、1.7 使用第20周期。

蓝色下横坐标：
    k = 1.4，St = 0.2、0.3、0.4、0.5、0.6；
    全部使用第20周期。

子图内容：
(a) C2（圆圈）和 C4（方块）
(b) Delta12
(c) 归一化总脉动动能 K'
(d) 模态熵 H

K' 默认在两条参数扫描线上分别除以各自最大值：
    固定 St=0.3 的 k 扫描单独归一化；
    固定 k=1.4 的 St 扫描单独归一化。
可通过 KPRIME_NORMALIZATION 修改为全局归一化。
"""

import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 用户配置
# ============================================================

SAVE_ROOT = r"E:\Rensq\DC_explore_F\POD_results\POD_results_3_1_1"
FIG_SAVE_DIR = r"E:\Rensq\DC_explore_F\POD_results\paper_figures"
os.makedirs(FIG_SAVE_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(SAVE_ROOT, "all_cases_pod_metrics_summary.csv")
OUT_BASENAME = "Fig_POD_dual_x_metrics_2x2"

# 固定 St=0.3，改变 k
K_SWEEP_ST = 0.3
K_SWEEP_VALUES = [0.8, 1.1, 1.4, 1.7]
K_SWEEP_CYCLES = {0.8: 15, 1.1: 15, 1.4: 20, 1.7: 20}

# 固定 k=1.4，改变 St
ST_SWEEP_K = 1.4
ST_SWEEP_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
ST_SWEEP_CYCLE = 20

DEFAULT_A = 0.0
DEFAULT_DC = 1.0

# 纵坐标范围
CUMULATIVE_YLIM = (0.5, 1.0)
DELTA12_YLIM = (0.0, 0.2)
NORMALIZED_KPRIME_YLIM = (0.0, 1.05)
H_YLIM = (0.0, 1.0)

# K' 归一化方式：
# "per_sweep"：两条扫描线分别除以自身最大值；
# "global"：两条扫描线统一除以所有工况中的最大值。
KPRIME_NORMALIZATION = "global"

FIGSIZE = (6.5, 5.25)
DPI = 600


# ============================================================
# 2. 绘图风格
# ============================================================

COLOR_K = "#D55E00"      # 橙色：固定 St=0.3，改变 k
COLOR_ST = "#0072B2"     # 蓝色：固定 k=1.4，改变 St
COLOR_BLACK = "#202020"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


# ============================================================
# 3. 数据读取
# ============================================================

def to_float(value):
    if value is None or str(value).strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_number(row, keys):
    for key in keys:
        if key in row:
            value = to_float(row.get(key))
            if np.isfinite(value):
                return value
    return np.nan


def read_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到汇总文件：{path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"汇总文件为空：{path}")

    print("CSV 字段：")
    print(", ".join(rows[0].keys()))
    return rows


def row_k(row):
    return first_number(row, ["k", "K", "reduced_frequency"])


def row_st(row):
    return first_number(row, ["St", "st", "ST", "strouhal"])


def row_a(row):
    return first_number(row, ["a", "A", "a0", "alpha", "Alpha"])


def row_dc(row):
    return first_number(row, ["dc", "DC", "duty_cycle"])


def row_cycle(row):
    """读取周期编号；也支持从 cycle_15、period_20 等文本中提取。"""
    keys = [
        "cycle", "Cycle", "cycle_id", "cycle_index", "cycle_number",
        "selected_cycle", "pod_cycle", "snapshot_cycle",
        "period", "Period", "period_id", "period_index", "period_number",
        "selected_period", "cycle_end", "end_cycle", "period_end", "end_period",
    ]

    value = first_number(row, keys)
    if np.isfinite(value):
        return int(round(value))

    pattern = re.compile(r"(?:cycle|period)[_\-=\s]*(\d+)", re.I)
    for key, value in row.items():
        key_lower = str(key).lower()
        if not any(word in key_lower for word in
                   ["cycle", "period", "path", "file", "name"]):
            continue

        match = pattern.search(f"{key} {value}")
        if match:
            return int(match.group(1))

    return None


def get_modal_energy_fractions(row):
    """
    从 E1、E2、... 列读取模态能量占比，并按模态编号排序。
    仅保留有限且大于零的值。
    """
    mode_energy = []
    pattern = re.compile(r"^E[_\-\s]?(\d+)$", re.I)

    for key, raw_value in row.items():
        match = pattern.match(str(key).strip())
        if not match:
            continue

        value = to_float(raw_value)
        if np.isfinite(value) and value > 0.0:
            mode_energy.append((int(match.group(1)), value))

    mode_energy.sort(key=lambda item: item[0])
    return np.asarray([value for _, value in mode_energy], dtype=float)


def calculate_normalized_modal_entropy(row):
    """
    由模态能量占比计算归一化模态熵：

        H = -sum(p_m ln p_m) / ln(N)

    其中 p_m 由全部可用 E_m 再归一化得到，N 为有效模态数。
    当有效模态少于两个时返回 NaN。
    """
    energies = get_modal_energy_fractions(row)
    if energies.size < 2:
        return np.nan

    total = float(np.sum(energies))
    if not np.isfinite(total) or total <= 0.0:
        return np.nan

    probabilities = energies / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / np.log(probabilities.size)


def metric(row, name):
    if name == "C2":
        value = first_number(row, ["C2", "c2", "C_2", "c_2"])
        if np.isfinite(value):
            return value

        e1 = first_number(row, ["E1", "e1", "E_1", "e_1"])
        e2 = first_number(row, ["E2", "e2", "E_2", "e_2"])
        return e1 + e2 if np.isfinite(e1) and np.isfinite(e2) else np.nan

    if name == "C4":
        value = first_number(row, ["C4", "c4", "C_4", "c_4"])
        if np.isfinite(value):
            return value

        values = [
            first_number(row, ["E1", "e1", "E_1", "e_1"]),
            first_number(row, ["E2", "e2", "E_2", "e_2"]),
            first_number(row, ["E3", "e3", "E_3", "e_3"]),
            first_number(row, ["E4", "e4", "E_4", "e_4"]),
        ]
        return float(np.sum(values)) if all(np.isfinite(values)) else np.nan

    if name == "Delta12":
        value = first_number(
            row,
            ["Delta12", "delta12", "Delta_12", "delta_12", "D12", "d12"],
        )
        if np.isfinite(value):
            return value

        e1 = first_number(row, ["E1", "e1", "E_1", "e_1"])
        e2 = first_number(row, ["E2", "e2", "E_2", "e_2"])
        return abs(e1 - e2) if np.isfinite(e1) and np.isfinite(e2) else np.nan

    if name == "Kprime":
        value = first_number(row, [
            "Kprime", "kprime", "K_prime", "k_prime", "K'", "Kp", "kp",
            "K_total", "k_total", "total_fluctuating_kinetic_energy",
            "fluctuating_kinetic_energy", "total_pod_energy",
            "POD_total_energy", "sum_lambda", "lambda_sum",
            "sum_eigenvalues", "eigenvalue_sum",
        ])
        if np.isfinite(value):
            return value

        # 若汇总表没有 K' 列，则尝试将全部特征值列相加
        eigenvalues = []
        pattern = re.compile(r"^(?:lambda|eig|eigenvalue)[_\-\s]*\d+$", re.I)
        for key, raw_value in row.items():
            if pattern.match(str(key).strip()):
                value = to_float(raw_value)
                if np.isfinite(value):
                    eigenvalues.append(value)

        return float(np.sum(eigenvalues)) if eigenvalues else np.nan

    if name == "H":
        value = first_number(row, [
            "H", "h", "modal_entropy", "Modal_entropy", "modalEntropy",
            "normalized_modal_entropy", "normalized_entropy",
            "entropy_H", "H_modal",
        ])
        if np.isfinite(value):
            return value

        return calculate_normalized_modal_entropy(row)

    raise ValueError(f"未知指标：{name}")


# ============================================================
# 4. 按工况和周期筛选
# ============================================================

def optional_match(value, target, tol=1.0e-9):
    return (not np.isfinite(value)) or abs(value - target) <= tol


def find_case(rows, k_target, st_target, cycle_target, tol=1.0e-9):
    candidates = []

    for row in rows:
        k_value = row_k(row)
        st_value = row_st(row)

        if not np.isfinite(k_value) or not np.isfinite(st_value):
            continue
        if abs(k_value - k_target) > tol:
            continue
        if abs(st_value - st_target) > tol:
            continue
        if not optional_match(row_a(row), DEFAULT_A, tol):
            continue
        if not optional_match(row_dc(row), DEFAULT_DC, tol):
            continue

        candidates.append(row)

    if not candidates:
        raise ValueError(
            f"未找到工况：k={k_target}, St={st_target}, cycle={cycle_target}"
        )

    rows_with_cycle = [row for row in candidates if row_cycle(row) is not None]
    if not rows_with_cycle:
        raise ValueError(
            f"已找到 k={k_target}, St={st_target}，但无法识别周期列。\n"
            "请确认 CSV 中包含 cycle、period、selected_cycle、cycle_end "
            "等字段，或路径字段中含有 cycle_15、cycle_20。"
        )

    for row in rows_with_cycle:
        if row_cycle(row) == int(cycle_target):
            return row

    available = sorted({row_cycle(row) for row in rows_with_cycle})
    raise ValueError(
        f"k={k_target}, St={st_target} 没有第 {cycle_target} 周期；"
        f"可用周期为：{available}"
    )


def case_metrics(rows, k_value, st_value, cycle_value):
    row = find_case(rows, k_value, st_value, cycle_value)
    result = {
        "C2": metric(row, "C2"),
        "C4": metric(row, "C4"),
        "Delta12": metric(row, "Delta12"),
        "Kprime": metric(row, "Kprime"),
        "H": metric(row, "H"),
    }

    missing = [key for key, value in result.items() if not np.isfinite(value)]
    if missing:
        raise ValueError(
            f"k={k_value}, St={st_value}, cycle={cycle_value} "
            f"缺少指标：{', '.join(missing)}"
        )
    return result


def extract_sweeps(rows):
    metric_names = ["C2", "C4", "Delta12", "Kprime", "H"]

    k_data = {name: [] for name in metric_names}
    k_data["cycle"] = []
    for k_value in K_SWEEP_VALUES:
        cycle_value = K_SWEEP_CYCLES[k_value]
        values = case_metrics(rows, k_value, K_SWEEP_ST, cycle_value)
        for name in metric_names:
            k_data[name].append(values[name])
        k_data["cycle"].append(cycle_value)

    st_data = {name: [] for name in metric_names}
    st_data["cycle"] = []
    for st_value in ST_SWEEP_VALUES:
        values = case_metrics(rows, ST_SWEEP_K, st_value, ST_SWEEP_CYCLE)
        for name in metric_names:
            st_data[name].append(values[name])
        st_data["cycle"].append(ST_SWEEP_CYCLE)

    for data in [k_data, st_data]:
        for name in metric_names:
            data[name] = np.asarray(data[name], dtype=float)

    return k_data, st_data


def normalize_kprime(k_values, st_values, mode="per_sweep"):
    k_values = np.asarray(k_values, dtype=float)
    st_values = np.asarray(st_values, dtype=float)

    if mode == "per_sweep":
        k_scale = np.nanmax(k_values)
        st_scale = np.nanmax(st_values)
    elif mode == "global":
        global_scale = np.nanmax(np.r_[k_values, st_values])
        k_scale = global_scale
        st_scale = global_scale
    else:
        raise ValueError(
            "KPRIME_NORMALIZATION 只能设置为 'per_sweep' 或 'global'。"
        )

    if not np.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError("固定 St=0.3 扫描中的 K' 最大值无效，无法归一化。")
    if not np.isfinite(st_scale) or st_scale <= 0.0:
        raise ValueError("固定 k=1.4 扫描中的 K' 最大值无效，无法归一化。")

    return k_values / k_scale, st_values / st_scale, k_scale, st_scale


# ============================================================
# 5. 绘图辅助函数
# ============================================================

def panel_label(ax, letter):
    """
    仅字母为斜体，左右括号保持正体。
    例如 letter='a' 时显示为 (a)，其中 a 为斜体。
    """
    ax.text(
        -0.13,
        1.16,
        rf"($\mathit{{{letter}}}$)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )


def make_dual_x_axis(ax):
    """
    ax：下方蓝色 St 坐标轴；
    ax_top：上方橙色 k 坐标轴。
    """
    ax_top = ax.twiny()

    x_st = np.arange(len(ST_SWEEP_VALUES), dtype=float)
    ax.set_xlim(-0.2, len(ST_SWEEP_VALUES) - 0.8)
    ax.set_xticks(x_st)
    ax.set_xticklabels([f"{value:.1f}" for value in ST_SWEEP_VALUES])
    ax.set_xlabel(r"$St$  ($k=1.4$)", labelpad=3)
    ax.xaxis.label.set_color(COLOR_ST)
    ax.tick_params(
        axis="x",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        colors=COLOR_ST,
        length=3,
        width=0.7,
    )
    ax.spines["bottom"].set_color(COLOR_ST)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["top"].set_visible(False)

    x_k = np.arange(len(K_SWEEP_VALUES), dtype=float)
    ax_top.set_xlim(-0.15, len(K_SWEEP_VALUES) - 0.85)
    ax_top.set_xticks(x_k)
    ax_top.set_xticklabels([f"{value:.1f}" for value in K_SWEEP_VALUES])
    ax_top.set_xlabel(r"$k$  ($St=0.3$)", labelpad=2.5)
    ax_top.xaxis.set_label_position("top")
    ax_top.xaxis.tick_top()
    ax_top.xaxis.label.set_color(COLOR_K)
    ax_top.tick_params(
        axis="x",
        bottom=False,
        top=True,
        labelbottom=False,
        labeltop=True,
        colors=COLOR_K,
        length=3,
        width=0.7,
    )
    ax_top.spines["top"].set_color(COLOR_K)
    ax_top.spines["top"].set_linewidth(0.9)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.patch.set_visible(False)
    ax_top.yaxis.set_visible(False)

    ax.tick_params(
        axis="y",
        left=True,
        right=True,
        colors="black",
        length=3,
        width=0.6,
    )
    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    return ax_top, x_k, x_st


def draw_line(ax, x, y, color, marker):
    ax.plot(
        x,
        y,
        color=color,
        linestyle="-",
        linewidth=1.15,
        marker=marker,
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.85,
        zorder=3,
    )


def add_c2_c4_legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=COLOR_BLACK,
            markeredgewidth=0.9,
            label=r"$C_2$",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="s",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=COLOR_BLACK,
            markeredgewidth=0.9,
            label=r"$C_4$",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        handlelength=0.8,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.35,
    )


# ============================================================
# 6. 保存图中数据
# ============================================================

def save_plot_data(k_data, st_data, k_scale, st_scale):
    path = os.path.join(FIG_SAVE_DIR, OUT_BASENAME + "_data.csv")
    fields = [
        "sweep",
        "k",
        "St",
        "cycle",
        "C2",
        "C4",
        "Delta12",
        "Kprime",
        "Kprime_normalized",
        "Kprime_normalization_scale",
        "H",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for i, k_value in enumerate(K_SWEEP_VALUES):
            writer.writerow({
                "sweep": "St=0.3, varying k",
                "k": k_value,
                "St": K_SWEEP_ST,
                "cycle": k_data["cycle"][i],
                "C2": k_data["C2"][i],
                "C4": k_data["C4"][i],
                "Delta12": k_data["Delta12"][i],
                "Kprime": k_data["Kprime"][i],
                "Kprime_normalized": k_data["Kprime_normalized"][i],
                "Kprime_normalization_scale": k_scale,
                "H": k_data["H"][i],
            })

        for i, st_value in enumerate(ST_SWEEP_VALUES):
            writer.writerow({
                "sweep": "k=1.4, varying St",
                "k": ST_SWEEP_K,
                "St": st_value,
                "cycle": st_data["cycle"][i],
                "C2": st_data["C2"][i],
                "C4": st_data["C4"][i],
                "Delta12": st_data["Delta12"][i],
                "Kprime": st_data["Kprime"][i],
                "Kprime_normalized": st_data["Kprime_normalized"][i],
                "Kprime_normalization_scale": st_scale,
                "H": st_data["H"][i],
            })

    return path


# ============================================================
# 7. 主程序
# ============================================================

def plot_pod_metrics():
    rows = read_rows(SUMMARY_CSV)
    k_data, st_data = extract_sweeps(rows)

    (
        k_data["Kprime_normalized"],
        st_data["Kprime_normalized"],
        k_scale,
        st_scale,
    ) = normalize_kprime(
        k_data["Kprime"],
        st_data["Kprime"],
        mode=KPRIME_NORMALIZATION,
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=FIGSIZE,
        constrained_layout=False,
    )

    # --------------------------------------------------------
    # (a) C2 和 C4
    # --------------------------------------------------------
    ax = axes[0, 0]
    ax_top, x_k, x_st = make_dual_x_axis(ax)

    draw_line(ax, x_st, st_data["C2"], COLOR_ST, "o")
    draw_line(ax, x_st, st_data["C4"], COLOR_ST, "s")
    draw_line(ax_top, x_k, k_data["C2"], COLOR_K, "o")
    draw_line(ax_top, x_k, k_data["C4"], COLOR_K, "s")

    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(CUMULATIVE_YLIM)
    ax.set_yticks(np.linspace(CUMULATIVE_YLIM[0], CUMULATIVE_YLIM[1], 6))
    add_c2_c4_legend(ax)
    panel_label(ax, "a")

    # --------------------------------------------------------
    # (b) Delta12
    # --------------------------------------------------------
    ax = axes[0, 1]
    ax_top, x_k, x_st = make_dual_x_axis(ax)

    draw_line(ax, x_st, st_data["Delta12"], COLOR_ST, "D")
    draw_line(ax_top, x_k, k_data["Delta12"], COLOR_K, "D")

    ax.set_ylabel(r"$\Delta_{12}$")
    delta_max = np.nanmax(np.r_[k_data["Delta12"], st_data["Delta12"]])
    ax.set_ylim(
        DELTA12_YLIM[0],
        max(DELTA12_YLIM[1], 1.15 * delta_max),
    )
    panel_label(ax, "b")

    # --------------------------------------------------------
    # (c) 归一化总脉动动能 K'
    # --------------------------------------------------------
    ax = axes[1, 0]
    ax_top, x_k, x_st = make_dual_x_axis(ax)

    draw_line(ax, x_st, st_data["Kprime_normalized"], COLOR_ST, "^")
    draw_line(ax_top, x_k, k_data["Kprime_normalized"], COLOR_K, "^")

    ax.set_ylabel(r"Normalized $K^\prime$")
    ax.set_ylim(NORMALIZED_KPRIME_YLIM)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    panel_label(ax, "c")

    # --------------------------------------------------------
    # (d) 模态熵 H；绘制形式与图(b)相同
    # --------------------------------------------------------
    ax = axes[1, 1]
    ax_top, x_k, x_st = make_dual_x_axis(ax)

    draw_line(ax, x_st, st_data["H"], COLOR_ST, "D")
    draw_line(ax_top, x_k, k_data["H"], COLOR_K, "D")

    ax.set_ylabel(r"Modal entropy $H$")
    ax.set_ylim(H_YLIM)
    ax.set_yticks(np.linspace(H_YLIM[0], H_YLIM[1], 6))
    panel_label(ax, "d")

    # 双横坐标需要较大的行间距，避免上、下两行标签重叠
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.115,
        top=0.900,
        wspace=0.38,
        hspace=0.66,
    )

    saved = []
    for extension in ["png", "pdf", "svg"]:
        path = os.path.join(FIG_SAVE_DIR, f"{OUT_BASENAME}.{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)
        saved.append(path)

    csv_path = save_plot_data(k_data, st_data, k_scale, st_scale)
    plt.close(fig)

    print("\n图像与数据已保存：")
    for path in saved:
        print(path)
    print(csv_path)

    print(f"\nK' 归一化方式：{KPRIME_NORMALIZATION}")
    print(f"固定 St=0.3 扫描的归一化尺度：{k_scale:.8g}")
    print(f"固定 k=1.4 扫描的归一化尺度：{st_scale:.8g}")

    print("\n固定 St=0.3 的 k 扫描：")
    print("k     cycle    C2        C4        Delta12   Kprime     Kprime*    H")
    for i, k_value in enumerate(K_SWEEP_VALUES):
        print(
            f"{k_value:<3.1f}   {k_data['cycle'][i]:<7d}  "
            f"{k_data['C2'][i]:<8.5f}  {k_data['C4'][i]:<8.5f}  "
            f"{k_data['Delta12'][i]:<8.5f}  {k_data['Kprime'][i]:<10.5g} "
            f"{k_data['Kprime_normalized'][i]:<9.5f}  {k_data['H'][i]:<8.5f}"
        )

    print("\n固定 k=1.4 的 St 扫描：")
    print("St    cycle    C2        C4        Delta12   Kprime     Kprime*    H")
    for i, st_value in enumerate(ST_SWEEP_VALUES):
        print(
            f"{st_value:<3.1f}   {st_data['cycle'][i]:<7d}  "
            f"{st_data['C2'][i]:<8.5f}  {st_data['C4'][i]:<8.5f}  "
            f"{st_data['Delta12'][i]:<8.5f}  {st_data['Kprime'][i]:<10.5g} "
            f"{st_data['Kprime_normalized'][i]:<9.5f}  {st_data['H'][i]:<8.5f}"
        )


if __name__ == "__main__":
    plot_pod_metrics()