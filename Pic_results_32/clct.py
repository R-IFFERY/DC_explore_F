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
    "legend.fontsize": 11,

    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble":
        r"\usepackage{amsmath} "
        r"\usepackage{newtxtext,newtxmath}",

    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # --------------------------------------------------------
    # 透明背景
    # --------------------------------------------------------
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
})


# ============================================================
# 路径设置
# ============================================================
data_root = (
    r"E:/Rensq/DC_explore_F/"
    r"results_aneq0/cycle_averaged_results_selected"
)

save_root = data_root

os.makedirs(
    save_root,
    exist_ok=True
)


# ============================================================
# 指定工况
# ============================================================
st = 0.3

case_dict = {
    0.8: [19, 20, 21],
    1.1: [19, 20, 21],
    1.4: [16, 17, 18],
    1.7: [10, 11, 12],
}


# ============================================================
# 不同 k 对应使用的周期范围
# ============================================================
period_range_dict = {
    0.8: (11, 15),
    1.1: (11, 15),
    1.4: (16, 20),
    1.7: (16, 20),
}


# ============================================================
# 绘制的系数列
# ============================================================
cl_col = "cl_total_mean"
ct_col = "ct_total_mean"


# ============================================================
# 不同 k 的绘图样式
#
# 所有曲线均为黑色；
# 不同 k 通过 marker + linestyle 区分。
# ============================================================
style_dict = {

    0.8: {
        "marker": "o",
        "linestyle": "-"
    },

    1.1: {
        "marker": "s",
        "linestyle": "--"
    },

    1.4: {
        "marker": "^",
        "linestyle": "-."
    },

    1.7: {
        "marker": "D",
        "linestyle": ":"
    },
}


# ============================================================
# 绘图参数
# ============================================================
LINEWIDTH = 0.8
MARKERSIZE = 4.0

CAPSIZE = 3.0
ERROR_LINEWIDTH = 0.8
MARKER_EDGE_WIDTH = 0.9


# ============================================================
# 数值转字符串
# ============================================================
def fmt(x):
    return f"{x:g}"


# ============================================================
# 汇总数据
# ============================================================
summary_rows = []

for k, a_list in case_dict.items():

    p_start, p_end = period_range_dict[k]

    for a in a_list:

        file_name = (
            f"k={fmt(k)}"
            f"st={fmt(st)}"
            f"a={fmt(a)}"
            f"dc=100.csv"
        )

        file_path = os.path.join(
            data_root,
            file_name
        )

        # ----------------------------------------------------
        # 检查文件
        # ----------------------------------------------------
        if not os.path.exists(file_path):

            print(
                f"缺少文件，跳过: "
                f"{file_path}"
            )

            continue

        # ----------------------------------------------------
        # 读取 CSV
        # ----------------------------------------------------
        df = pd.read_csv(
            file_path
        )

        required_cols = [
            "period",
            cl_col,
            ct_col
        ]

        missing_cols = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:

            print(
                f"文件 {file_name} "
                f"缺少列 {missing_cols}，跳过。"
            )

            continue

        # ----------------------------------------------------
        # 选取指定周期
        # ----------------------------------------------------
        df_sel = df[
            (df["period"] >= p_start)
            &
            (df["period"] <= p_end)
        ].copy()

        if df_sel.empty:

            print(
                f"文件 {file_name} "
                f"在周期 {p_start}-{p_end} "
                f"内无数据，跳过。"
            )

            continue

        # ----------------------------------------------------
        # 计算 CL 均值和样本标准差
        # ----------------------------------------------------
        cl_mean = (
            df_sel[
                cl_col
            ].mean()
        )

        cl_std = (
            df_sel[
                cl_col
            ].std(
                ddof=1
            )
        )

        # ----------------------------------------------------
        # 计算 CT 均值和样本标准差
        # ----------------------------------------------------
        ct_mean = (
            df_sel[
                ct_col
            ].mean()
        )

        ct_std = (
            df_sel[
                ct_col
            ].std(
                ddof=1
            )
        )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------
        summary_rows.append({

            "k": k,
            "st": st,
            "a": a,

            "period_start":
                p_start,

            "period_end":
                p_end,

            "cl_mean":
                cl_mean,

            "cl_std":
                cl_std,

            "ct_mean":
                ct_mean,

            "ct_std":
                ct_std,

            "n_periods":
                len(df_sel),
        })


# ============================================================
# 转换为 DataFrame
# ============================================================
summary_df = pd.DataFrame(
    summary_rows
)

if summary_df.empty:

    raise ValueError(
        "没有读取到任何有效数据，"
        "请检查 data_root 路径和文件名。"
    )


# ============================================================
# 保存汇总数据
# ============================================================
summary_csv = os.path.join(
    save_root,
    "summary_cycle_mean_for_plot.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False,
    encoding="utf-8-sig",
    float_format="%.8e"
)

print(
    f"汇总数据已保存: "
    f"{summary_csv}"
)


# ============================================================
# 绘图
#
# 左图 = CT
# 右图 = CL
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(6.5, 2.8),
    sharex=False
)

# ------------------------------------------------------------
# 整张图背景透明
# ------------------------------------------------------------
fig.patch.set_facecolor(
    "none"
)

fig.patch.set_alpha(
    0.0
)

ax1, ax2 = axes


# ============================================================
# 绘制不同 k
# ============================================================
for k in case_dict.keys():

    df_k = summary_df[
        summary_df["k"] == k
    ].copy()

    if df_k.empty:
        continue

    df_k = df_k.sort_values(
        "a"
    )

    x = (
        df_k[
            "a"
        ].values
    )

    style = style_dict[k]

    # --------------------------------------------------------
    # 左图：推力系数 CT
    # --------------------------------------------------------
    ax1.errorbar(

        x,
        df_k[
            "ct_mean"
        ].values,

        yerr=df_k[
            "ct_std"
        ].values,

        # ----------------------------------------------------
        # 不同 k 的 marker + linestyle
        # ----------------------------------------------------
        marker=style[
            "marker"
        ],

        linestyle=style[
            "linestyle"
        ],

        # ----------------------------------------------------
        # 全部黑色
        # ----------------------------------------------------
        color="black",
        ecolor="black",

        # ----------------------------------------------------
        # 空心数据点
        # ----------------------------------------------------
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=MARKER_EDGE_WIDTH,

        # ----------------------------------------------------
        # 曲线和误差棒
        # ----------------------------------------------------
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,

        capsize=CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,

        label=rf"$k={fmt(k)}$",

        zorder=3
    )

    # --------------------------------------------------------
    # 右图：升力系数 CL
    # --------------------------------------------------------
    ax2.errorbar(

        x,
        df_k[
            "cl_mean"
        ].values,

        yerr=df_k[
            "cl_std"
        ].values,

        # ----------------------------------------------------
        # 不同 k 的 marker + linestyle
        # ----------------------------------------------------
        marker=style[
            "marker"
        ],

        linestyle=style[
            "linestyle"
        ],

        # ----------------------------------------------------
        # 全部黑色
        # ----------------------------------------------------
        color="black",
        ecolor="black",

        # ----------------------------------------------------
        # 空心数据点
        # ----------------------------------------------------
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=MARKER_EDGE_WIDTH,

        # ----------------------------------------------------
        # 曲线和误差棒
        # ----------------------------------------------------
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,

        capsize=CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,

        zorder=3
    )


# ============================================================
# 坐标轴标签
# ============================================================
ax1.set_xlabel(
    r"$\alpha_0/^\circ$"
)

ax1.set_ylabel(
    r"$\overline{C}_{TS}$",
    rotation=0
)


ax2.set_xlabel(
    r"$\alpha_0/^\circ$"
)

ax2.set_ylabel(
    r"$\overline{C}_{LS}$",
    rotation=0
)


# ============================================================
# 水平纵坐标标签位置
# ============================================================
ax1.yaxis.set_label_coords(
    -0.13,
    0.5
)

ax2.yaxis.set_label_coords(
    -0.13,
    0.5
)


# ============================================================
# y 轴范围
# ============================================================
ax1.set_ylim(
    0,
    0.3
)

ax2.set_ylim(
    1,
    2.5
)


# ============================================================
# CT 图增加 y=0.2 水平虚线
# ============================================================
ax1.axhline(
    y=0.2,
    linestyle="--",
    linewidth=0.8,
    color="black",
    zorder=1
)


# ============================================================
# 坐标轴外观设置
# ============================================================
for ax in axes:

    # --------------------------------------------------------
    # 子图背景完全透明
    # --------------------------------------------------------
    ax.set_facecolor(
        "none"
    )

    ax.patch.set_facecolor(
        "none"
    )

    ax.patch.set_alpha(
        0.0
    )

    # --------------------------------------------------------
    # 四条边框
    # --------------------------------------------------------
    ax.spines[
        "top"
    ].set_visible(
        True
    )

    ax.spines[
        "right"
    ].set_visible(
        True
    )

    ax.spines[
        "left"
    ].set_visible(
        True
    )

    ax.spines[
        "bottom"
    ].set_visible(
        True
    )

    # --------------------------------------------------------
    # 刻度
    # --------------------------------------------------------
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True
    )

    ax.grid(
        False
    )


# ============================================================
# 图例
#
# 仅左图显示
# ============================================================
ax1.legend(
    frameon=False
)


# ============================================================
# 横坐标刻度
# ============================================================
all_a = sorted(
    summary_df[
        "a"
    ].unique()
)

ax1.set_xticks(
    all_a
)

ax2.set_xticks(
    all_a
)


# ============================================================
# 子图外部左上角添加 (a) 和 (b)
# ============================================================
ax1.text(
    -0.24,
    1.03,
    r"($a$)",

    transform=ax1.transAxes,

    ha="left",
    va="bottom"
)


ax2.text(
    -0.24,
    1.03,
    r"($b$)",

    transform=ax2.transAxes,

    ha="left",
    va="bottom"
)


# ============================================================
# 调整布局
# ============================================================
plt.tight_layout()


# ============================================================
# 保存图像
# ============================================================
fig_png = os.path.join(
    save_root,
    "cycle_mean_CT_CL_vs_alpha0.png"
)

fig_svg = os.path.join(
    save_root,
    "cycle_mean_CT_CL_vs_alpha0.svg"
)


# ============================================================
# PNG
# ============================================================
fig.savefig(
    fig_png,
    dpi=600,
    bbox_inches="tight",
    transparent=True,
    facecolor="none",
    edgecolor="none"
)


# ============================================================
# SVG
# ============================================================
fig.savefig(
    fig_svg,
    bbox_inches="tight",
    transparent=True,
    facecolor="none",
    edgecolor="none"
)


plt.close(
    fig
)


# ============================================================
# 完成
# ============================================================
print(
    f"图像已保存: "
    f"{fig_png}"
)

print(
    f"图像已保存: "
    f"{fig_svg}"
)