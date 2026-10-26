# -*- coding: utf-8 -*-
"""
Plot DC = 0.25 big-period local correlation maps
for four propulsion-balanced k cases from existing NPZ caches.

Panels:
(a) k = 0.8, St = 0.3, alpha_0 = 20 deg
(b) k = 1.1, St = 0.3, alpha_0 = 20 deg
(c) k = 1.4, St = 0.3, alpha_0 = 17 deg
(d) k = 1.7, St = 0.3, alpha_0 = 11 deg

Correlation:
cycles 21-24 vs cycles 25-28

DC = 0.25
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Figure style
# ============================================================

USE_TEX = True
FONT_SIZE = 10

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,

    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "text.usetex": USE_TEX,

    "axes.linewidth": 0.7,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble": (
            r"\usepackage{amsmath}"
            r"\usepackage{newtxtext,newtxmath}"
        )
    })


# ============================================================
# User settings
# ============================================================

ST = 0.3
DC = 0.25

# 推进平衡角
CASE_DICT = {
    0.8: 20,
    1.1: 20,
    1.4: 17,
    1.7: 11,
}

K_LIST = [
    0.8,
    1.1,
    1.4,
    1.7,
]


# ============================================================
# Paths
# ============================================================

SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/flow/"
    r"local_correlation_DC"
)

FIGURE_BASENAME = os.path.join(
    SAVE_ROOT,
    "DC0p25_k0p8_1p1_1p4_1p7_rho_big_1x4"
)


# ============================================================
# Plot window
# ============================================================

# 图框完整显示：
# -10 <= y/c <= 10

X_MIN = -1.0
X_MAX = 20.0

Y_MIN = -5.0
Y_MAX = 5.0


# 但实际相关性场只绘制：
# -9 <= y/c <= 9
#
# 外侧 [-10,-9) 和 (9,10] 设置为 NaN，
# 与你给出的旧代码处理方式一致。

Y_DRAW_MIN = -9.0
Y_DRAW_MAX = 9.0


# ============================================================
# Utilities
# ============================================================

def fmt_num(value):
    """
    0.8  -> 0.8
    1.0  -> 1
    20.0 -> 20
    """
    return f"{float(value):g}"


def fmt_num_short(value):
    """
    用于目录和文件名：

    0.8  -> 0p8
    0.25 -> 0p25
    """
    return (
        fmt_num(value)
        .replace(".", "p")
        .replace("-", "m")
    )


def get_npz_path(k, st, a, dc):
    """
    根据原 big-period correlation 代码的保存规则构造 NPZ 路径。

    例如：

    SAVE_ROOT/
        k0p8_St0p3_a20/
            k0p8_St0p3_a20_dc0p25_
            rho_big_21-24_vs_25-28.npz
    """

    case_dir = os.path.join(
        SAVE_ROOT,
        (
            f"k{fmt_num_short(k)}"
            f"_St{fmt_num_short(st)}"
            f"_a{fmt_num_short(a)}"
        )
    )

    filename = (
        f"k{fmt_num_short(k)}"
        f"_St{fmt_num_short(st)}"
        f"_a{fmt_num_short(a)}"
        f"_dc{fmt_num_short(dc)}"
        f"_rho_big_21-24_vs_25-28.npz"
    )

    return os.path.join(
        case_dir,
        filename
    )


def load_rho_npz(npz_path):
    """
    从已有 NPZ 中读取：
        x/c
        y/c
        rho_big
    """

    if not os.path.isfile(npz_path):

        raise FileNotFoundError(
            "Cannot find NPZ file:\n"
            f"{npz_path}"
        )

    with np.load(
        npz_path,
        allow_pickle=False
    ) as data:

        required = {
            "x_over_c",
            "y_over_c",
            "rho_map",
        }

        missing = required.difference(
            data.files
        )

        if missing:

            raise KeyError(
                f"NPZ file lacks required arrays: "
                f"{sorted(missing)}\n"
                f"{npz_path}"
            )

        X = np.array(
            data["x_over_c"],
            dtype=float
        )

        Y = np.array(
            data["y_over_c"],
            dtype=float
        )

        rho_map = np.array(
            data["rho_map"],
            dtype=float
        )

    return (
        X,
        Y,
        rho_map
    )


def mask_boundary_rows(
    Y,
    field,
    y_draw_min,
    y_draw_max
):
    """
    图框仍显示 -10 <= y/c <= 10，

    但仅保留：
        y_draw_min <= y/c <= y_draw_max

    其余区域设为 NaN。
    """

    field_masked = np.array(
        field,
        dtype=float,
        copy=True
    )

    keep = (
        (Y >= y_draw_min)
        &
        (Y <= y_draw_max)
    )

    field_masked[
        ~keep
    ] = np.nan

    return field_masked


# ============================================================
# Main plotting
# ============================================================

def main():

    # ========================================================
    # 1. Load four cases
    # ========================================================

    loaded_cases = []

    for k in K_LIST:

        a = CASE_DICT[k]

        npz_path = get_npz_path(
            k=k,
            st=ST,
            a=a,
            dc=DC
        )

        X, Y, rho_map = load_rho_npz(
            npz_path
        )

        # ----------------------------------------------------
        # 与你给出的旧代码保持一致：
        #
        # 图框显示 y/c = -10 ~ 10，
        # 但只绘制 y/c = -9 ~ 9。
        # ----------------------------------------------------

        rho_map = mask_boundary_rows(
            Y,
            rho_map,
            Y_DRAW_MIN,
            Y_DRAW_MAX
        )

        loaded_cases.append({

            "k": k,

            "st": ST,

            "a": a,

            "dc": DC,

            "X": X,

            "Y": Y,

            "rho_map": rho_map,

            "npz_path": npz_path,
        })

    # ========================================================
    # 2. Figure
    # ========================================================

    fig, axes = plt.subplots(

        1,
        4,

        figsize=(
            6.5,
            1.2
        ),

        dpi=300,

        sharex=True,
        sharey=True,

        squeeze=False,
    )

    flat_axes = axes.ravel()


    # ========================================================
    # 3. Colormap
    # ========================================================

    cmap = plt.get_cmap(
        "RdBu_r"
    ).copy()

    # 与给出的代码一致：
    # 无效区域 / 被屏蔽区域为白色
    cmap.set_bad(
        "white"
    )

    norm = Normalize(
        vmin=-1.0,
        vmax=1.0
    )


    # ========================================================
    # 4. Plot four panels
    # ========================================================

    mesh = None

    for panel_index, case in enumerate(
        loaded_cases
    ):

        ax = flat_axes[
            panel_index
        ]

        # ----------------------------------------------------
        # Correlation map
        # ----------------------------------------------------

        mesh = ax.pcolormesh(

            case["X"],

            case["Y"],

            np.ma.masked_invalid(
                case["rho_map"]
            ),

            cmap=cmap,

            norm=norm,

            shading="auto",

            # 与你给出的代码保持一致：
            # SVG 中不栅格化
            rasterized=True,
        )

        # ----------------------------------------------------
        # Axis
        # ----------------------------------------------------

        ax.set_xlim(
            X_MIN,
            X_MAX
        )

        ax.set_ylim(
            Y_MIN,
            Y_MAX
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.tick_params(
            top=True,
            right=True
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(
            r"$x/c$"
        )

        if panel_index == 0:

            ax.set_ylabel(
                r"$y/c$"
            )

        # ----------------------------------------------------
        # Title
        #
        # St 均为 0.3，因此这里只标 k。
        # 如果想和旧图完全一样显示 k, St，
        # 可以换成下面注释掉的版本。
        # ----------------------------------------------------
        #
        # ax.set_title(
        #
        #     rf"$k={case['k']:g}$",
        #
        #     pad=3.0,
        # )

        # 如果希望同时写 St：
        #
        # ax.set_title(
        #     rf"$k={case['k']:g},\ "
        #     rf"St={case['st']:g}$",
        #     pad=3.0,
        # )

        # ----------------------------------------------------
        # Panel label
        #
        # (a) (b) (c) (d)
        #
        # 与给出的代码保持同样位置
        # ----------------------------------------------------

        ax.text(

            -0.10,
            0.98,

            rf"$({chr(97 + panel_index)})$",

            transform=ax.transAxes,

            ha="left",

            va="bottom",

            clip_on=False,
        )


    # ========================================================
    # 5. Inset colorbar
    #
    # 放在 panel (d) 内部右上角
    # ========================================================

    ax_d = flat_axes[3]

    cax = inset_axes(

        ax_d,

        width="40%",

        height="5%",

        loc="upper right",

        borderpad=0.55,
    )

    cbar = fig.colorbar(

        mesh,

        cax=cax,

        orientation="horizontal",

        ticks=[
            -1.0,
            0.0,
            1.0
        ],
    )

    # 保持色条为矢量
    if (
        hasattr(
            cbar,
            "solids"
        )
        and
        cbar.solids is not None
    ):

        cbar.solids.set_rasterized(
            False
        )

    cbar.outline.set_linewidth(
        0.6
    )

    cbar.ax.tick_params(

        length=2.0,

        width=0.6,

        pad=1.0
    )


    # ========================================================
    # 6. Layout
    # ========================================================

    fig.subplots_adjust(

        left=0.067,

        right=0.992,

        bottom=0.180,

        top=0.900,

        wspace=0.10,

        hspace=0.0,
    )


    # ========================================================
    # 7. Save
    # ========================================================

    png_path = (
        FIGURE_BASENAME
        + ".png"
    )

    svg_path = (
        FIGURE_BASENAME
        + ".svg"
    )

    # pdf_path = (
    #     FIGURE_BASENAME
    #     + ".pdf"
    # )

    fig.savefig(

        png_path,

        dpi=600,

        bbox_inches="tight"
    )

    fig.savefig(

        svg_path,

        bbox_inches="tight"
    )

    # fig.savefig(
    #
    #     pdf_path,
    #
    #     bbox_inches="tight"
    # )

    plt.close(
        fig
    )


    # ========================================================
    # 8. Information
    # ========================================================

    print(
        "Selected DC=0.25 cases "
        "plotted successfully."
    )

    for case in loaded_cases:

        finite = case[
            "rho_map"
        ][
            np.isfinite(
                case["rho_map"]
            )
        ]

        if finite.size > 0:

            mean_rho = np.mean(
                finite
            )

            print(

                f"  k={case['k']:g}, "
                f"St={case['st']:g}, "
                f"a={case['a']:g}, "
                f"DC={case['dc']:g}, "
                f"mean rho={mean_rho:.6f}"

            )

        else:

            print(

                f"  k={case['k']:g}, "
                f"St={case['st']:g}, "
                f"a={case['a']:g}, "
                f"DC={case['dc']:g}, "
                f"no valid rho"

            )

        print(
            f"    <- {case['npz_path']}"
        )

    print(
        f"Saved PNG: {png_path}"
    )

    print(
        f"Saved SVG: {svg_path}"
    )

    # print(
    #     f"Saved PDF: {pdf_path}"
    # )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    main()