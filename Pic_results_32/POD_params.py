# -*- coding: utf-8 -*-
"""
plot_pod_metrics_C2C4_H_vs_k_selected_angles.py

读取 pod_311_periodic_batch.py 生成的 all_cases_pod_metrics_summary.csv，
仅绘制以下 4 个工况：

    k=0.8, St=0.3, a=20
    k=1.1, St=0.3, a=20
    k=1.4, St=0.3, a=17
    k=1.7, St=0.3, a=11

采用 k 为横轴，绘制：
    (a) C2 和 C4
    (b) H

说明：
- 直接读取汇总 CSV 中的 C2、C4、H，不在绘图阶段重算；
- 严格检查周期：
    k=0.8、1.1 -> cycle_start=cycle_end=15
    k=1.4、1.7 -> cycle_start=cycle_end=20
- 输出 PNG、PDF、SVG 及绘图数据 CSV。
"""

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ============================================================
# 1. 用户配置
# ============================================================

SAVE_ROOT = r"E:\Rensq\DC_explore_F\POD_results\POD_results_3_1_2_aneq0"
SUMMARY_CSV = os.path.join(SAVE_ROOT, "all_cases_pod_metrics_summary.csv")

FIG_SAVE_DIR = os.path.join(SAVE_ROOT, "paper_figures")
os.makedirs(FIG_SAVE_DIR, exist_ok=True)

OUT_BASENAME = "Fig_POD_C2C4_H_vs_k_selected_angles"

# 仅选择这 4 个工况
SELECTED_CASES: List[Tuple[float, float, float]] = [
    (0.8, 0.3, 20),
    (1.1, 0.3, 20),
    (1.4, 0.3, 17),
    (1.7, 0.3, 11),
]

CYCLE_BY_K = {
    0.8: (15, 15),
    1.1: (15, 15),
    1.4: (20, 20),
    1.7: (20, 20),
}

TARGET_DC = 1.0
MATCH_TOL = 1.0e-9

CUMULATIVE_YLIM = (0.50, 1.00)
H_YLIM = (0.00, 1.00)

FIGSIZE = (6.5, 2.85)
DPI = 600
SAVE_TRANSPARENT = True


# ============================================================
# 2. 绘图风格
# ============================================================

COLOR_C2 = "#0072B2"
COLOR_C4 = "#D55E00"
COLOR_H = "#009E73"
COLOR_BLACK = "#202020"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.transparent": SAVE_TRANSPARENT,
})


# ============================================================
# 3. 数据读取
# ============================================================

REQUIRED_COLUMNS = {
    "k", "St", "a", "dc", "cycle_start", "cycle_end", "C2", "C4", "H"
}


def to_float(value, field_name: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"字段 {field_name!r} 为空。")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"字段 {field_name!r} 的值 {value!r} 不能转换为浮点数。"
        ) from exc

    if not np.isfinite(number):
        raise ValueError(f"字段 {field_name!r} 的值不是有限数：{value!r}")
    return number


def same_number(value: float, target: float, tol: float = MATCH_TOL) -> bool:
    return abs(value - target) <= tol


def read_summary_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 POD 汇总文件：\n{path}\n"
            "请先运行 POD 计算代码并确认 SAVE_ROOT 设置一致。"
        )

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                "POD 汇总文件缺少绘图所需字段：" + ", ".join(missing_columns)
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"POD 汇总文件为空：{path}")

    return rows


def parse_row(row: Dict[str, str], row_index: int) -> Dict[str, float]:
    parsed = {
        "k": to_float(row.get("k"), "k"),
        "St": to_float(row.get("St"), "St"),
        "a": to_float(row.get("a"), "a"),
        "dc": to_float(row.get("dc"), "dc"),
        "cycle_start": to_float(row.get("cycle_start"), "cycle_start"),
        "cycle_end": to_float(row.get("cycle_end"), "cycle_end"),
        "C2": to_float(row.get("C2"), "C2"),
        "C4": to_float(row.get("C4"), "C4"),
        "H": to_float(row.get("H"), "H"),
        "source_row": row_index,
        "case_name": row.get("case_name", ""),
    }

    if parsed["C2"] < -MATCH_TOL or parsed["C2"] > 1.0 + MATCH_TOL:
        raise ValueError(f"第 {row_index} 行 C2={parsed['C2']} 超出范围 [0,1]。")
    if parsed["C4"] < -MATCH_TOL or parsed["C4"] > 1.0 + MATCH_TOL:
        raise ValueError(f"第 {row_index} 行 C4={parsed['C4']} 超出范围 [0,1]。")
    if parsed["C4"] + MATCH_TOL < parsed["C2"]:
        raise ValueError(
            f"第 {row_index} 行出现 C4 < C2：C2={parsed['C2']}, C4={parsed['C4']}。"
        )
    if parsed["H"] < -MATCH_TOL or parsed["H"] > 1.0 + MATCH_TOL:
        raise ValueError(f"第 {row_index} 行 H={parsed['H']} 超出范围 [0,1]。")

    return parsed


def find_exact_case(
    parsed_rows: List[Dict[str, float]],
    k_target: float,
    st_target: float,
    a_target: float,
) -> Dict[str, float]:
    expected_start, expected_end = CYCLE_BY_K[k_target]

    same_case = [
        row for row in parsed_rows
        if same_number(row["k"], k_target)
        and same_number(row["St"], st_target)
        and same_number(row["a"], a_target)
        and same_number(row["dc"], TARGET_DC)
    ]

    if not same_case:
        raise ValueError(
            f"未找到工况：k={k_target}, St={st_target}, a={a_target}, dc={TARGET_DC}。"
        )

    exact_cycle = [
        row for row in same_case
        if same_number(row["cycle_start"], expected_start)
        and same_number(row["cycle_end"], expected_end)
    ]

    if not exact_cycle:
        available = sorted({
            (int(round(row["cycle_start"])), int(round(row["cycle_end"])))
            for row in same_case
        })
        raise ValueError(
            f"工况 k={k_target}, St={st_target}, a={a_target} 应使用周期 "
            f"{expected_start}-{expected_end}，但可用周期为 {available}。"
        )

    if len(exact_cycle) > 1:
        row_ids = [int(row["source_row"]) for row in exact_cycle]
        raise ValueError(
            f"工况 k={k_target}, St={st_target}, a={a_target} 存在重复记录，行号 {row_ids}。"
        )

    return exact_cycle[0]


def extract_selected_data(rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    parsed_rows = [parse_row(row, i + 2) for i, row in enumerate(rows)]
    selected = []

    for k_value, st_value, a_value in SELECTED_CASES:
        row = find_exact_case(parsed_rows, float(k_value), float(st_value), float(a_value))
        selected.append(row)

    selected.sort(key=lambda item: item["k"])
    return selected


# ============================================================
# 4. 绘图辅助
# ============================================================


def panel_label(ax, letter: str):
    ax.text(
        -0.16,
        1.04,
        rf"($\mathit{{{letter}}}$)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )


def set_horizontal_ylabel(ax, label: str, x: float = -0.16):
    ax.set_ylabel(label, rotation=0, labelpad=0)
    ax.yaxis.set_label_coords(x, 0.5)
    ax.yaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("center")


def style_axis(ax):
    ax.tick_params(axis="both", which="both", top=True, right=True, length=3, width=0.6)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)


def draw_line(ax, x, y, color, marker, linestyle="-"):
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=1.15,
        marker=marker,
        markersize=4.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.9,
        zorder=3,
    )


# ============================================================
# 5. 保存图中数据
# ============================================================


def save_plot_data(data: List[Dict[str, float]]) -> str:
    path = os.path.join(FIG_SAVE_DIR, OUT_BASENAME + "_data.csv")
    fields = [
        "case_name", "k", "St", "a", "dc", "cycle_start", "cycle_end", "C2", "C4", "H"
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in data:
            writer.writerow({field: item[field] for field in fields})

    return path


# ============================================================
# 6. 主程序
# ============================================================


def plot_pod_metrics():
    rows = read_summary_rows(SUMMARY_CSV)
    data = extract_selected_data(rows)

    k_values = np.asarray([item["k"] for item in data], dtype=float)
    c2 = np.asarray([item["C2"] for item in data], dtype=float)
    c4 = np.asarray([item["C4"] for item in data], dtype=float)
    entropy = np.asarray([item["H"] for item in data], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=False)
    ax_cumulative, ax_entropy = axes

    # (a) C2 和 C4
    draw_line(ax_cumulative, k_values, c2, color=COLOR_C2, marker="o", linestyle="-")
    draw_line(ax_cumulative, k_values, c4, color=COLOR_C4, marker="s", linestyle="--")
    ax_cumulative.set_xlabel(r"$k$")
    set_horizontal_ylabel(ax_cumulative, r"$C_i$")
    ax_cumulative.set_ylim(CUMULATIVE_YLIM)
    ax_cumulative.set_yticks(np.linspace(CUMULATIVE_YLIM[0], CUMULATIVE_YLIM[1], 6))
    ax_cumulative.set_xticks(k_values)
    ax_cumulative.set_xticklabels([f"{value:.1f}" for value in k_values])
    style_axis(ax_cumulative)
    panel_label(ax_cumulative, "a")
    ax_cumulative.legend(
        handles=[
            Line2D([0], [0], color=COLOR_C2, linestyle="-", linewidth=1.15,
                   marker="o", markersize=4.8, markerfacecolor="white",
                   markeredgecolor=COLOR_C2, markeredgewidth=0.9, label=r"$C_2$"),
            Line2D([0], [0], color=COLOR_C4, linestyle="--", linewidth=1.15,
                   marker="s", markersize=4.8, markerfacecolor="white",
                   markeredgecolor=COLOR_C4, markeredgewidth=0.9, label=r"$C_4$"),
        ],
        frameon=False,
        loc="lower right",
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.35,
    )

    # (b) H
    draw_line(ax_entropy, k_values, entropy, color=COLOR_H, marker="D", linestyle="-")
    ax_entropy.set_xlabel(r"$k$")
    set_horizontal_ylabel(ax_entropy, r"$H$")
    ax_entropy.set_ylim(H_YLIM)
    ax_entropy.set_yticks(np.linspace(H_YLIM[0], H_YLIM[1], 6))
    ax_entropy.set_xticks(k_values)
    ax_entropy.set_xticklabels([f"{value:.1f}" for value in k_values])
    style_axis(ax_entropy)
    panel_label(ax_entropy, "b")
    ax_entropy.legend(
        handles=[
            Line2D([0], [0], color=COLOR_H, linestyle="-", linewidth=1.15,
                   marker="D", markersize=4.8, markerfacecolor="white",
                   markeredgecolor=COLOR_H, markeredgewidth=0.9, label=r"$H$"),
        ],
        frameon=False,
        loc="best",
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.35,
    )

    x_min = np.min(k_values) - 0.08
    x_max = np.max(k_values) + 0.08
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.92, wspace=0.34)

    saved_paths = []
    for extension in ["png", "pdf", "svg"]:
        path = os.path.join(FIG_SAVE_DIR, f"{OUT_BASENAME}.{extension}")
        save_kwargs = {"bbox_inches": "tight", "transparent": SAVE_TRANSPARENT}
        if extension == "png":
            save_kwargs["dpi"] = DPI
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)

    csv_path = save_plot_data(data)
    plt.close(fig)

    print("\n图像与绘图数据已保存：")
    for path in saved_paths:
        print(path)
    print(csv_path)

    print("\n实际读取的数据：")
    print("k     St    a      cycle      C2         C4         H")
    for item in data:
        cycle_text = f"{int(round(item['cycle_start']))}-{int(round(item['cycle_end']))}"
        print(
            f"{item['k']:<3.1f}   {item['St']:<3.1f}   {item['a']:<5g}  "
            f"{cycle_text:<9s}  {item['C2']:<10.6f} {item['C4']:<10.6f} {item['H']:<10.6f}"
        )


if __name__ == "__main__":
    plot_pod_metrics()