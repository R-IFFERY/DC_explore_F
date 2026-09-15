# -*- coding: utf-8 -*-
"""
plot_fig_pod_dc_H_C2_C4.py

针对新的 DC 大周期 POD 结果绘图。

数据：
    St = 0.3

    k = 0.8, 1.1, 1.4, 1.7

对应平均俯仰角：
    k=0.8 -> alpha0=20 deg
    k=1.1 -> alpha0=20 deg
    k=1.4 -> alpha0=17 deg
    k=1.7 -> alpha0=11 deg

DC：
    1.00, 0.75, 0.50, 0.25

子图：
    (a) Modal entropy H
    (b) C2 and C4

横坐标：
    DC

不同 k：
    使用不同颜色

C2 / C4：
    C2 -> 圆形 marker + 实线
    C4 -> 方形 marker + 虚线

图例：
    不同 k 的图例放在 (a) 子图内部，
    图例仅显示彩色线段，不显示 marker。
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. 用户配置
# ============================================================

# 新的大周期 POD 结果目录
SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/"
    r"POD_results/POD_333_intermittent"
)

FIG_SAVE_DIR = (
    r"E:/Rensq/DC_explore_F/"
    r"POD_results/POD_333_intermittent/paper_figures_dc"
)

os.makedirs(
    FIG_SAVE_DIR,
    exist_ok=True
)


# ============================================================
# 汇总 CSV
# ============================================================

SUMMARY_CSV = os.path.join(
    SAVE_ROOT,
    "all_cases_pod_metrics_summary.csv"
)

OUT_BASENAME = (
    "Fig_POD_DC_H_C2_C4"
)


# ============================================================
# 工况
# ============================================================

ST_TARGET = 0.3

K_VALUES = [
    0.8,
    1.1,
    1.4,
    1.7,
]


# k 对应后翼平均俯仰角
K_A_MAP = {
    0.8: 20,
    1.1: 20,
    1.4: 17,
    1.7: 11,
}


# ============================================================
# DC
#
# 从左到右：
# 1 -> 0.75 -> 0.5 -> 0.25
# ============================================================

DC_VALUES = [
    1.00,
    0.75,
    0.50,
    0.25,
]


# ============================================================
# Y 轴范围
# ============================================================

# Modal entropy
H_YLIM = (
    0.0,
    1.0
)

H_YTICKS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]


# C2 / C4
CUMULATIVE_YLIM = (
    0.0,
    1.0
)

CUMULATIVE_YTICKS = [
    0,
    # 0.1,
    0.2,
    # 0.3,
    0.4,
    # 0.5,
    0.6,
    # 0.7,
    0.8,
    # 0.9,
    1.0
]


# ============================================================
# Figure
# ============================================================

FIGSIZE = (
    6.5,
    2.65
)

DPI = 600


# ============================================================
# 背景透明
# ============================================================

TRANSPARENT_BACKGROUND = True


# ============================================================
# 2. 绘图颜色
#
# 不同 k 使用不同颜色
# ============================================================

K_COLORS = {
    0.8: "#0072B2",
    1.1: "#D55E00",
    1.4: "#009E73",
    1.7: "#CC79A7",
}


# ============================================================
# 曲线设置
# ============================================================

LINE_WIDTH = 1.15

MARKER_SIZE = 4.5

MARKER_EDGE_WIDTH = 0.85


# H
H_MARKER = "D"

H_LINESTYLE = "-"


# C2
C2_MARKER = "o"

C2_LINESTYLE = "-"


# C4
C4_MARKER = "s"

C4_LINESTYLE = "--"


# ============================================================
# 3. Matplotlib style
# ============================================================

plt.rcParams.update({

    "font.size": 10,

    "axes.titlesize": 10,

    "axes.labelsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    "legend.fontsize": 9,

    "font.family": "serif",

    "font.serif": [
        "Times New Roman"
    ],

    "text.usetex": True,

    "pgf.texsystem": "pdflatex",

    "text.latex.preamble": (
        r"\usepackage{amsmath} "
        r"\usepackage{newtxtext,newtxmath}"
    ),

    "axes.linewidth": 0.7,

    "xtick.direction": "in",

    "ytick.direction": "in",

    "figure.facecolor": "none",

    "axes.facecolor": "none",

    "savefig.facecolor": "none",

    "savefig.edgecolor": "none",

    # SVG 保持文字而不是转换为路径
    "svg.fonttype": "none",
})


# ============================================================
# 4. 数据读取工具
# ============================================================

def to_float(value):

    if (
        value is None
        or
        str(value).strip() == ""
    ):

        return np.nan

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return np.nan


def first_number(
    row,
    keys
):

    for key in keys:

        if key in row:

            value = to_float(
                row.get(
                    key
                )
            )

            if np.isfinite(
                value
            ):

                return value

    return np.nan


# ============================================================
# 读取 summary CSV
# ============================================================

def read_rows(
    path
):

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"未找到 POD 汇总文件：\n"
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    if not rows:

        raise ValueError(
            f"汇总 CSV 为空：{path}"
        )

    print(
        "\nCSV 字段："
    )

    print(
        ", ".join(
            rows[0].keys()
        )
    )

    return rows


# ============================================================
# 字段读取
# ============================================================

def row_k(
    row
):

    return first_number(
        row,
        [
            "k",
            "K",
            "reduced_frequency"
        ]
    )


def row_st(
    row
):

    return first_number(
        row,
        [
            "St",
            "st",
            "ST",
            "strouhal"
        ]
    )


def row_a(
    row
):

    return first_number(
        row,
        [
            "a",
            "A",
            "a0",
            "alpha",
            "Alpha"
        ]
    )


def row_dc(
    row
):

    return first_number(
        row,
        [
            "dc",
            "DC",
            "duty_cycle"
        ]
    )


# ============================================================
# 5. POD 指标
# ============================================================

def get_modal_energy_fractions(
    row
):
    """
    从 E1、E2、... 中读取全部模态能量。
    """

    energies = []

    i = 1

    while True:

        possible_keys = [
            f"E{i}",
            f"e{i}",
            f"E_{i}",
            f"e_{i}"
        ]

        found = False

        for key in possible_keys:

            if key in row:

                value = to_float(
                    row[
                        key
                    ]
                )

                if np.isfinite(
                    value
                ):

                    energies.append(
                        value
                    )

                found = True

                break

        if not found:
            break

        i += 1

    return np.asarray(
        energies,
        dtype=float
    )


# ============================================================
# H fallback
# ============================================================

def calculate_normalized_modal_entropy(
    row
):
    """
    如果 CSV 没有 H，则由 E_i 重新计算：

        H = -sum(p_i ln p_i) / ln(N)
    """

    energies = (
        get_modal_energy_fractions(
            row
        )
    )

    energies = energies[
        np.isfinite(
            energies
        )
        &
        (
            energies > 0
        )
    ]

    if energies.size < 2:

        return np.nan

    total = np.sum(
        energies
    )

    if total <= 0:

        return np.nan

    p = (
        energies
        /
        total
    )

    entropy = -np.sum(
        p
        *
        np.log(
            p
        )
    )

    return float(
        entropy
        /
        np.log(
            p.size
        )
    )


# ============================================================
# 读取 C2 / C4 / H
# ============================================================

def metric(
    row,
    name
):

    # ========================================================
    # C2
    # ========================================================

    if name == "C2":

        value = first_number(
            row,
            [
                "C2",
                "c2",
                "C_2",
                "c_2"
            ]
        )

        if np.isfinite(
            value
        ):

            return value

        E1 = first_number(
            row,
            [
                "E1",
                "e1",
                "E_1",
                "e_1"
            ]
        )

        E2 = first_number(
            row,
            [
                "E2",
                "e2",
                "E_2",
                "e_2"
            ]
        )

        if (
            np.isfinite(E1)
            and
            np.isfinite(E2)
        ):

            return (
                E1
                +
                E2
            )

        return np.nan


    # ========================================================
    # C4
    # ========================================================

    if name == "C4":

        value = first_number(
            row,
            [
                "C4",
                "c4",
                "C_4",
                "c_4"
            ]
        )

        if np.isfinite(
            value
        ):

            return value

        values = []

        for i in range(
            1,
            5
        ):

            value_i = first_number(
                row,
                [
                    f"E{i}",
                    f"e{i}",
                    f"E_{i}",
                    f"e_{i}"
                ]
            )

            values.append(
                value_i
            )

        if all(
            np.isfinite(
                values
            )
        ):

            return float(
                np.sum(
                    values
                )
            )

        return np.nan


    # ========================================================
    # H
    # ========================================================

    if name == "H":

        value = first_number(
            row,
            [
                "H",
                "h",
                "modal_entropy",
                "normalized_modal_entropy",
                "normalized_entropy",
                "entropy_H"
            ]
        )

        if np.isfinite(
            value
        ):

            return value

        return (
            calculate_normalized_modal_entropy(
                row
            )
        )

    raise ValueError(
        f"未知 POD 指标：{name}"
    )


# ============================================================
# 6. 查找一个 k-DC 工况
# ============================================================

def find_case(
    rows,
    k_target,
    dc_target,
    tol=1.0e-8
):
    """
    当前新的 summary CSV 每个 k-DC
    应只有一个大周期 POD 结果。

    因此不再按 cycle 筛选。
    """

    expected_a = (
        K_A_MAP[
            k_target
        ]
    )

    candidates = []

    for row in rows:

        k_value = row_k(
            row
        )

        st_value = row_st(
            row
        )

        dc_value = row_dc(
            row
        )

        a_value = row_a(
            row
        )

        # ----------------------------------------------------
        # k
        # ----------------------------------------------------

        if not np.isfinite(
            k_value
        ):

            continue

        if abs(
            k_value
            -
            k_target
        ) > tol:

            continue


        # ----------------------------------------------------
        # St = 0.3
        # ----------------------------------------------------

        if not np.isfinite(
            st_value
        ):

            continue

        if abs(
            st_value
            -
            ST_TARGET
        ) > tol:

            continue


        # ----------------------------------------------------
        # DC
        # ----------------------------------------------------

        if not np.isfinite(
            dc_value
        ):

            continue

        if abs(
            dc_value
            -
            dc_target
        ) > tol:

            continue


        # ----------------------------------------------------
        # alpha0
        #
        # 若 summary 中有 a，则检查；
        # 若没有 a，则不因此排除。
        # ----------------------------------------------------

        if np.isfinite(
            a_value
        ):

            if abs(
                a_value
                -
                expected_a
            ) > tol:

                continue

        candidates.append(
            row
        )

    if not candidates:

        raise ValueError(
            f"未找到工况："
            f"k={k_target}, "
            f"St={ST_TARGET}, "
            f"a={expected_a}, "
            f"DC={dc_target}"
        )

    if len(
        candidates
    ) > 1:

        print(
            f"警告："
            f"k={k_target}, "
            f"DC={dc_target} "
            f"找到 {len(candidates)} 行，"
            f"默认使用第一行。"
        )

    return candidates[
        0
    ]


# ============================================================
# 7. 提取全部 k-DC 数据
# ============================================================

def extract_data(
    rows
):

    data = {}

    for k in K_VALUES:

        data[k] = {

            "dc": [],

            "H": [],

            "C2": [],

            "C4": [],

            "selected_cycles": [],

            "n_snapshots": [],
        }

        for dc in DC_VALUES:

            row = find_case(
                rows,
                k_target=k,
                dc_target=dc
            )

            H = metric(
                row,
                "H"
            )

            C2 = metric(
                row,
                "C2"
            )

            C4 = metric(
                row,
                "C4"
            )

            # =================================================
            # 数据检查
            # =================================================

            for name, value in [
                ("H", H),
                ("C2", C2),
                ("C4", C4)
            ]:

                if not np.isfinite(
                    value
                ):

                    raise ValueError(
                        f"k={k}, "
                        f"DC={dc} "
                        f"缺少指标 {name}"
                    )

            data[k][
                "dc"
            ].append(
                dc
            )

            data[k][
                "H"
            ].append(
                H
            )

            data[k][
                "C2"
            ].append(
                C2
            )

            data[k][
                "C4"
            ].append(
                C4
            )


            # -------------------------------------------------
            # 记录 POD 使用周期
            # -------------------------------------------------

            selected_cycles = row.get(
                "selected_cycles",
                ""
            )

            data[k][
                "selected_cycles"
            ].append(
                selected_cycles
            )


            # -------------------------------------------------
            # snapshots
            # -------------------------------------------------

            n_snapshots = first_number(
                row,
                [
                    "n_snapshots",
                    "snapshots"
                ]
            )

            data[k][
                "n_snapshots"
            ].append(
                n_snapshots
            )


        # =====================================================
        # convert to ndarray
        # =====================================================

        for key in [
            "dc",
            "H",
            "C2",
            "C4",
            "n_snapshots"
        ]:

            data[k][key] = np.asarray(
                data[k][key],
                dtype=float
            )

    return data


# ============================================================
# 8. Panel label
# ============================================================

def panel_label(
    ax,
    letter
):
    """
    括号正体、字母斜体。
    """

    ax.text(
        -0.14,
        1.04,

        rf"($\mathit{{{letter}}}$)",

        transform=ax.transAxes,

        ha="left",

        va="bottom",

        fontsize=10,

        clip_on=False
    )


# ============================================================
# 9. 坐标轴
# ============================================================

def setup_dc_axis(
    ax
):

    # --------------------------------------------------------
    # 从左到右：
    #
    # DC=1 -> 0.75 -> 0.5 -> 0.25
    # --------------------------------------------------------

    ax.set_xlim(
        1.05,
        0.20
    )

    ax.set_xticks([
        1.00,
        0.75,
        0.50,
        0.25
    ])

    ax.set_xticklabels([
        r"$1$",
        r"$0.75$",
        r"$0.5$",
        r"$0.25$"
    ])

    ax.set_xlabel(
        r"$DC$"
    )

    ax.tick_params(
        axis="both",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.6
    )

    # 水平辅助线
    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.35
    )

    ax.set_axisbelow(
        True
    )


# ============================================================
# 10. 普通 k 曲线
# ============================================================

def draw_k_line(
    ax,
    x,
    y,
    k,
    marker,
    linestyle="-",
    label=None
):

    color = (
        K_COLORS[
            k
        ]
    )

    ax.plot(
        x,
        y,

        color=color,

        linestyle=linestyle,

        linewidth=LINE_WIDTH,

        marker=marker,

        markersize=MARKER_SIZE,

        markerfacecolor="white",

        markeredgecolor=color,

        markeredgewidth=MARKER_EDGE_WIDTH,

        label=label,

        zorder=3
    )


# ============================================================
# 11. k legend
#
# 只显示不同颜色的线段，不显示 marker
# ============================================================

def make_k_legend_handles():

    handles = []

    for k in K_VALUES:

        handles.append(

            Line2D(
                [0],
                [0],

                color=K_COLORS[k],

                linestyle="-",

                linewidth=LINE_WIDTH,

                marker=None,

                label=rf"$k={k:g}$"
            )
        )

    return handles


# ============================================================
# 12. C2 / C4 legend
# ============================================================

def add_c2_c4_legend(
    ax
):

    handles = [

        Line2D(
            [0],
            [0],

            color="#202020",

            linestyle=C2_LINESTYLE,

            linewidth=LINE_WIDTH,

            marker=C2_MARKER,

            markersize=5,

            markerfacecolor="white",

            markeredgecolor="#202020",

            label=r"$C_2$"
        ),

        Line2D(
            [0],
            [0],

            color="#202020",

            linestyle=C4_LINESTYLE,

            linewidth=LINE_WIDTH,

            marker=C4_MARKER,

            markersize=5,

            markerfacecolor="white",

            markeredgecolor="#202020",

            label=r"$C_4$"
        ),
    ]

    legend = ax.legend(
        handles=handles,

        frameon=False,

        loc="lower right",

        handlelength=1.7,

        handletextpad=0.4,

        labelspacing=0.3,

        borderaxespad=0.35
    )

    return legend


# ============================================================
# 13. 保存图中数据
# ============================================================

def save_plot_data(
    data
):

    path = os.path.join(
        FIG_SAVE_DIR,
        OUT_BASENAME
        +
        "_data.csv"
    )

    fields = [
        "k",
        "St",
        "alpha0",
        "DC",
        "selected_cycles",
        "n_snapshots",
        "H",
        "C2",
        "C4",
    ]

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for k in K_VALUES:

            for i, dc in enumerate(
                DC_VALUES
            ):

                writer.writerow({

                    "k":
                        k,

                    "St":
                        ST_TARGET,

                    "alpha0":
                        K_A_MAP[k],

                    "DC":
                        dc,

                    "selected_cycles":
                        data[k][
                            "selected_cycles"
                        ][i],

                    "n_snapshots":
                        data[k][
                            "n_snapshots"
                        ][i],

                    "H":
                        data[k][
                            "H"
                        ][i],

                    "C2":
                        data[k][
                            "C2"
                        ][i],

                    "C4":
                        data[k][
                            "C4"
                        ][i],
                })

    return path


# ============================================================
# 14. 主绘图程序
# ============================================================

def plot_pod_metrics():

    # ========================================================
    # Read
    # ========================================================

    rows = read_rows(
        SUMMARY_CSV
    )

    data = extract_data(
        rows
    )


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=FIGSIZE,
        constrained_layout=False
    )


    # ========================================================
    # (a) Modal entropy H
    # ========================================================

    ax = axes[
        0
    ]

    setup_dc_axis(
        ax
    )

    for k in K_VALUES:

        draw_k_line(
            ax,

            data[k][
                "dc"
            ],

            data[k][
                "H"
            ],

            k=k,

            marker=H_MARKER,

            linestyle=H_LINESTYLE
        )

    ax.set_ylabel(
        r"$H$",
        rotation=0,
        labelpad=8
    )

    ax.set_ylim(
        H_YLIM
    )

    ax.set_yticks(
        H_YTICKS
    )

    panel_label(
        ax,
        "a"
    )


    # ========================================================
    # k legend
    #
    # 放在 (a) 子图内部
    # 仅显示彩色线，不显示 marker
    # ========================================================

    k_handles = make_k_legend_handles()

    ax.legend(
        handles=k_handles,

        loc="lower left",

        ncol=2,

        frameon=False,

        handlelength=1.8,

        handletextpad=0.4,

        columnspacing=0.8,

        labelspacing=0.3,

        borderaxespad=0.4
    )


    # ========================================================
    # (b) C2 / C4
    # ========================================================

    ax = axes[
        1
    ]

    setup_dc_axis(
        ax
    )

    for k in K_VALUES:

        # ----------------------------------------------------
        # C2
        # ----------------------------------------------------

        draw_k_line(
            ax,

            data[k][
                "dc"
            ],

            data[k][
                "C2"
            ],

            k=k,

            marker=C2_MARKER,

            linestyle=C2_LINESTYLE
        )


        # ----------------------------------------------------
        # C4
        # ----------------------------------------------------

        draw_k_line(
            ax,

            data[k][
                "dc"
            ],

            data[k][
                "C4"
            ],

            k=k,

            marker=C4_MARKER,

            linestyle=C4_LINESTYLE
        )

    ax.set_ylabel(
        r"$C_i$",
        rotation=0,
        labelpad=8
    )

    ax.set_ylim(
        CUMULATIVE_YLIM
    )

    ax.set_yticks(
        CUMULATIVE_YTICKS
    )

    add_c2_c4_legend(
        ax
    )

    panel_label(
        ax,
        "b"
    )


    # ========================================================
    # Layout
    #
    # 原来底部放置全局 k 图例，
    # 现在 k 图例已经移入 (a)，
    # 因此适当减小 bottom。
    # ========================================================

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.19,
        top=0.92,
        wspace=0.32
    )


    # ========================================================
    # 保存图中数据
    # ========================================================

    csv_path = (
        save_plot_data(
            data
        )
    )


    # ========================================================
    # Save figure
    # ========================================================

    saved_paths = []


    # PNG
    png_path = os.path.join(
        FIG_SAVE_DIR,
        OUT_BASENAME
        +
        ".png"
    )

    fig.savefig(
        png_path,

        dpi=DPI,

        bbox_inches="tight",

        transparent=TRANSPARENT_BACKGROUND,

        facecolor=(
            "none"
            if TRANSPARENT_BACKGROUND
            else "white"
        ),

        edgecolor="none"
    )

    saved_paths.append(
        png_path
    )


    # SVG
    svg_path = os.path.join(
        FIG_SAVE_DIR,
        OUT_BASENAME
        +
        ".svg"
    )

    fig.savefig(
        svg_path,

        format="svg",

        bbox_inches="tight",

        transparent=TRANSPARENT_BACKGROUND,

        facecolor=(
            "none"
            if TRANSPARENT_BACKGROUND
            else "white"
        ),

        edgecolor="none"
    )

    saved_paths.append(
        svg_path
    )


    # # PDF
    # pdf_path = os.path.join(
    #     FIG_SAVE_DIR,
    #     OUT_BASENAME
    #     +
    #     ".pdf"
    # )
    #
    # fig.savefig(
    #     pdf_path,
    #
    #     format="pdf",
    #
    #     bbox_inches="tight",
    #
    #     transparent=TRANSPARENT_BACKGROUND
    # )
    #
    # saved_paths.append(
    #     pdf_path
    # )
    #
    #
    # plt.close(
    #     fig
    # )


    # ========================================================
    # Print
    # ========================================================

    print(
        "\n"
        "=" * 72
    )

    print(
        "POD DC 指标图已完成"
    )

    print(
        "=" * 72
    )

    print(
        "\n图像："
    )

    for path in saved_paths:

        print(
            path
        )

    print(
        "\n图中数据："
    )

    print(
        csv_path
    )


    # ========================================================
    # 数据打印
    # ========================================================

    print(
        "\n"
        "POD metrics:"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        "k      DC      H          C2         C4"
    )

    print(
        "------------------------------------------------------------"
    )

    for k in K_VALUES:

        for i, dc in enumerate(
            DC_VALUES
        ):

            print(
                f"{k:<4.1f}   "
                f"{dc:<5.2f}   "
                f"{data[k]['H'][i]:<10.6f} "
                f"{data[k]['C2'][i]:<10.6f} "
                f"{data[k]['C4'][i]:<10.6f}"
            )

    print(
        "------------------------------------------------------------"
    )


# ============================================================
# 15. Run
# ============================================================

if __name__ == "__main__":

    plot_pod_metrics()