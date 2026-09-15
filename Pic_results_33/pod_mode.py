# -*- coding: utf-8 -*-
"""
plot_fig_pod_dc_big_period.py

读取新的 POD_DC_big_period 计算结果进行论文绘图。

数据来源：
    POD_DC_big_period

工况：
    St = 0.3

    k=0.8, alpha0=20 deg
    k=1.1, alpha0=20 deg
    k=1.4, alpha0=17 deg
    k=1.7, alpha0=11 deg

    DC = 1.00, 0.75, 0.50, 0.25

每个 DC 单独生成一张图。

每个 k 对应一个主面板：

第一行：
    Mode 1
    Mode 2
    Mode 3
    Mode 4

第二行：
    a1-a4 随整个 4T 大周期的变化
    a1-a2 phase portrait
    a3-a4 phase portrait
    a1-a3 phase portrait

重要：
新的 POD 本身已经针对一个完整 4T 大周期计算，
因此本程序不再从 POD 时间系数中提取某一个普通周期。

全部 128 个 snapshots 都参与绘图：

    tau/(4T) = 0 ~ 1

其中：

    0.00~0.25 : 第1个 T
    0.25~0.50 : 第2个 T
    0.50~0.75 : 第3个 T
    0.75~1.00 : 第4个 T
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import colormaps
from matplotlib.transforms import Bbox


# ============================================================
# 1. 用户配置区
# ============================================================

# ============================================================
# 新 POD 结果路径
# ============================================================

SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/"
    r"POD_results/POD_333_intermittent/"
)

FIG_SAVE_DIR = (
    r"E:/Rensq/DC_explore_F/"
    r"POD_results/POD_333_intermittent/paper_figures_dc/"
)

os.makedirs(
    FIG_SAVE_DIR,
    exist_ok=True
)


# ============================================================
# 工况设置
# ============================================================

ST = 0.3

K_A_MAP = {
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

DC_LIST = [
    1.00,
    0.75,
    0.50,
    0.25,
]


# ============================================================
# 生成全部16个工况
# ============================================================

ALL_CASES = []

for dc in DC_LIST:

    for k in K_LIST:

        ALL_CASES.append({

            "k":
                k,

            "st":
                ST,

            "a":
                K_A_MAP[k],

            "dc":
                dc,
        })


# ============================================================
# POD 模态设置
# ============================================================

N_MODES = 4


# ============================================================
# 模态涡量加权
#
# sqrt_lambda:
#
#     sqrt(lambda_i) * omega_i
#
# ============================================================

MODAL_VORTICITY_WEIGHT = (
    "sqrt_lambda"
)

FALLBACK_TO_SQRT_ENERGY = True


# ============================================================
# 流场显示范围
# ============================================================

X_LIM = (
    -1.0,
    15.0
)

Y_LIM = (
    -3.0,
    3.0
)

X_TICKS = [
    0,
    5,
    10,
    15
]

Y_TICKS = [
    -3,
    0,
    3
]


# ============================================================
# 模态图归一化
#
# global:
#
# 所有
#   4 k
# × 4 DC
# × 4 modes
#
# 共用同一个归一化尺度。
#
# 因此不同 DC 图之间也能比较模态幅值。
# ============================================================

NORMALISE_BY = (
    "global"
)

NORMALISE_PERCENTILE = 99.0


# ============================================================
# Color map
# ============================================================

CMAP_VMIN = -1.0
CMAP_VMAX = 1.0

WHITE_SIGMA = 0.16

SHOW_MASK_CONTOUR = False


# ============================================================
# 大周期时间系数
# ============================================================

NORMALISE_TIME_COEFFS = False

COEFF_LINE_WIDTH = 0.9

COEFF_MARKER_SIZE = 2.0

COEFF_SHOW_MARKERS = False


# ------------------------------------------------------------
# 如果新 POD 的 ai 幅值发生明显变化，
# 可以将 COEFF_Y_LIM 设置为 None，
# 自动根据全部数据计算范围。
# ------------------------------------------------------------

COEFF_Y_LIM = (
    -150.0,
    150.0
)

COEFF_Y_TICKS = [
    -150.0,
    0.0,
    150.0
]


# 大周期横坐标
COEFF_X_TICKS = [
    0.0,
    0.5,
    1.0
]

COEFF_X_TICK_LABELS = [
    r"$0$",
    r"$0.5$",
    r"$1$"
]


# ============================================================
# 是否画普通 T 周期边界
# ============================================================

SHOW_SUBCYCLE_DIVIDERS = True

SUBCYCLE_DIVIDERS = [
    0.25,
    0.50,
    0.75
]


# ============================================================
# 时间系数图图例
# ============================================================

COEFF_LEGEND_NCOL = 4

COEFF_LEGEND_LOC = (
    "lower center"
)

COEFF_LEGEND_BBOX = (
    0.5,
    -0.03
)


# ============================================================
# Phase portrait
# ============================================================

NORMALISE_PHASE_COEFFS = False

PHASE_AXIS_MIN = -150.0

PHASE_AXIS_MAX = 150.0

PHASE_TICKS = [
    -150,
    0,
    150
]

PHASE_LINE_WIDTH = 1.0

PHASE_SHOW_ZERO_LINES = True

CLOSE_PHASE_PORTRAIT = True


# ============================================================
# 字体
# ============================================================

FONT_SIZE = 10.0

ENERGY_LABEL_FONT_SIZE = 10

INNER_PANEL_LABEL_FONT_SIZE = 10

MAIN_PANEL_LABEL_FONT_SIZE = 10.0


# ============================================================
# Label padding
# ============================================================

AXIS_LABEL_PAD = -1.3

PHASE_AXIS_LABEL_PAD = -1

COEFF_Y_LABEL_PAD = -8.0

PHASE_Y_LABEL_PAD = -8.0


# ============================================================
# Figure size / layout
# ============================================================

FIG_WIDTH = 6.5

MARGIN_LEFT_INCH = 0.62

MARGIN_RIGHT_INCH = 0.18

MARGIN_TOP_INCH = 0.50

MARGIN_BOTTOM_INCH = 0.48

COLUMN_GAP_INCH = 0.085


# 第二行四张图之间距离
BOTTOM_ROW_EQUAL_GAP_INCH = 0.5


# 第二行水平移动
BOTTOM_ROW_CENTER_SHIFT_INCH = -0.5


# 上下排距离
TOP_BOTTOM_ROW_GAP_INCH = 0.30


# 不同 k 主面板之间距离
CASE_BLOCK_GAP_INCH = 0.40


# 第二行高度
BOTTOM_ROW_HEIGHT_SCALE = 0.4


# ============================================================
# Colorbar
# ============================================================

SHOW_INSET_COLORBAR = True

INSET_COLORBAR_BOUNDS = [
    0.57,
    0.22,
    0.36,
    0.065
]

COLORBAR_TICKS = [
    -1.0,
    0.0,
    1.0
]

COLORBAR_TICK_LABEL_SIZE = 7.5


# ============================================================
# 输出设置
# ============================================================

OUT_PREFIX = (
    "Fig_POD_St0.3_DC"
)

DPI = 600


# ============================================================
# Transparent background
# ============================================================

TRANSPARENT_SVG_BACKGROUND = True

TRANSPARENT_MASKED_REGIONS = True


# ============================================================
# 2. Matplotlib style
# ============================================================

plt.rcParams.update({

    "font.size":
        FONT_SIZE,

    "axes.titlesize":
        FONT_SIZE,

    "axes.labelsize":
        FONT_SIZE,

    "xtick.labelsize":
        FONT_SIZE,

    "ytick.labelsize":
        FONT_SIZE,

    "legend.fontsize":
        8.0,

    "font.family":
        "serif",

    "font.serif":
        [
            "Times New Roman"
        ],

    "text.usetex":
        True,

    "pgf.texsystem":
        "pdflatex",

    "text.latex.preamble":
        (
            r"\usepackage{amsmath} "
            r"\usepackage{newtxtext,newtxmath}"
        ),

    "axes.linewidth":
        0.7,

    "xtick.direction":
        "in",

    "ytick.direction":
        "in",

    "figure.facecolor":
        "none",

    "axes.facecolor":
        "none",

    "savefig.facecolor":
        "none",

    "savefig.edgecolor":
        "none",

    "svg.fonttype":
        "none",
})


# ============================================================
# Colormap
# ============================================================

def build_white_center_cmap(
    vmin=-1.0,
    vmax=1.0,
    n=512,
    sigma=0.16
):
    """
    中心偏白的 RdBu_r。
    """

    base = (
        colormaps[
            "RdBu_r"
        ].resampled(
            n
        )
    )

    new_colors = base(
        np.linspace(
            0,
            1,
            n
        )
    )

    vals = np.linspace(
        vmin,
        vmax,
        n
    )

    weights = np.exp(
        -0.5
        *
        (vals / sigma) ** 2
    )

    for i in range(n):

        w = weights[i]

        new_colors[i] = (
            (1.0 - w)
            *
            new_colors[i]
            +
            w
            *
            np.array([
                1,
                1,
                1,
                1
            ])
        )

    cmap = (
        mcolors.ListedColormap(
            new_colors
        )
    )

    if TRANSPARENT_MASKED_REGIONS:

        cmap.set_bad(
            (
                1.0,
                1.0,
                1.0,
                0.0
            )
        )

    else:

        cmap.set_bad(
            "white"
        )

    return cmap


CMAP = build_white_center_cmap(

    CMAP_VMIN,

    CMAP_VMAX,

    sigma=
        WHITE_SIGMA
)


# ============================================================
# 3. Naming
# ============================================================

def fmt_num(
    x: float
) -> str:
    """
    必须与新的 POD_DC_big_period
    程序中的命名保持一致。

    1.0  -> 1
    0.8  -> 0.8
    0.50 -> 0.5
    """

    if float(x).is_integer():

        return str(
            int(x)
        )

    return f"{x:g}"


def fmt_num_short(
    x: float
) -> str:

    return (
        fmt_num(x)
        .replace(
            ".",
            "p"
        )
        .replace(
            "-",
            "m"
        )
    )


def case_name_from_values(

    k: float,

    st: float,

    a: float,

    dc: float

) -> str:

    return (

        f"k{fmt_num_short(k)}"

        f"_St{fmt_num_short(st)}"

        f"_a{fmt_num_short(a)}"

        f"_dc{fmt_num_short(dc)}"
    )


def get_npz_path(
    case: dict
) -> str:

    cname = (
        case_name_from_values(

            case[
                "k"
            ],

            case[
                "st"
            ],

            case[
                "a"
            ],

            case[
                "dc"
            ]
        )
    )

    return os.path.join(

        SAVE_ROOT,

        cname,

        f"{cname}_pod_results.npz"
    )


def get_labels_path(
    case: dict
) -> str:

    cname = (
        case_name_from_values(

            case[
                "k"
            ],

            case[
                "st"
            ],

            case[
                "a"
            ],

            case[
                "dc"
            ]
        )
    )

    return os.path.join(

        SAVE_ROOT,

        cname,

        f"{cname}_snapshot_labels.csv"
    )


# ============================================================
# 内部小图标签
# ============================================================

def roman_panel_label(
    index: int
) -> str:

    roman = [
        "i",
        "ii",
        "iii",
        "iv",
        "v",
        "vi",
        "vii",
        "viii"
    ]

    if not (
        0
        <= index
        < len(roman)
    ):

        raise ValueError(
            "内部编号只支持0-7。"
        )

    return (
        rf"$\mathrm{{({roman[index]})}}$"
    )


def add_inner_panel_label(
    ax,
    index: int
):

    ax.text(

        0.97,
        0.94,

        roman_panel_label(
            index
        ),

        transform=
            ax.transAxes,

        ha=
            "right",

        va=
            "top",

        fontsize=
            INNER_PANEL_LABEL_FONT_SIZE,

        zorder=
            20,

        bbox=dict(
            facecolor="none",
            edgecolor="none",
            alpha=0.0,
            pad=0.8
        )
    )


# ============================================================
# 4. Snapshot labels
# ============================================================

def load_snapshot_labels(

    case: dict,

    n_snapshots: int

) -> list:
    """
    读取新的大周期标签。

    主要使用：
        big_phase_frac
        local_cycle_id
        cycle_no
        phase_frac
    """

    labels_path = (
        get_labels_path(
            case
        )
    )

    if not os.path.exists(
        labels_path
    ):

        print(
            f"警告：未找到标签："
            f"{labels_path}"
        )

        return []

    labels = []

    with open(

        labels_path,

        "r",

        encoding=
            "utf-8-sig",

        newline=""

    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            try:

                labels.append({

                    "snapshot_index":
                        int(
                            row[
                                "snapshot_index"
                            ]
                        ),

                    "cycle_no":
                        int(
                            row[
                                "cycle_no"
                            ]
                        ),

                    "phase_frac":
                        float(
                            row[
                                "phase_frac"
                            ]
                        ),

                    "local_cycle_id":
                        int(
                            row[
                                "local_cycle_id"
                            ]
                        ),

                    "big_phase_frac":
                        float(
                            row[
                                "big_phase_frac"
                            ]
                        ),
                })

            except (
                KeyError,
                TypeError,
                ValueError
            ):

                continue

    labels.sort(

        key=lambda item:
            item[
                "snapshot_index"
            ]
    )

    if len(
        labels
    ) != n_snapshots:

        print(

            f"警告：标签数 "
            f"{len(labels)} "
            f"与 coeffs 数 "
            f"{n_snapshots} "
            f"不一致。"
        )

        return []

    return labels


# ============================================================
# 5. 读取单个 POD
# ============================================================

def load_case_pod_result(
    case: dict
) -> dict:

    npz_path = (
        get_npz_path(
            case
        )
    )

    if not os.path.exists(
        npz_path
    ):

        raise FileNotFoundError(

            f"未找到 POD 文件：\n"
            f"{npz_path}"
        )

    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    with np.load(
        npz_path
    ) as data:

        required_keys = [

            "X",

            "Y",

            "modes_w",

            "energy",

            "coeffs",

            "fluid_mask"
        ]

        for key in required_keys:

            if key not in data:

                raise KeyError(

                    f"{npz_path} "
                    f"中缺少 {key}"
                )

        X = np.asarray(
            data[
                "X"
            ]
        )

        Y = np.asarray(
            data[
                "Y"
            ]
        )

        modes_w = np.asarray(

            data[
                "modes_w"
            ],

            dtype=float
        )

        energy = np.asarray(

            data[
                "energy"
            ],

            dtype=float
        )

        eigvals = (

            np.asarray(

                data[
                    "eigvals"
                ],

                dtype=float
            )

            if "eigvals"
            in data

            else None
        )

        coeffs = np.asarray(

            data[
                "coeffs"
            ],

            dtype=float
        )

        fluid_mask = np.asarray(

            data[
                "fluid_mask"
            ],

            dtype=bool
        )

        # ----------------------------------------------------
        # 新 POD NPZ 中直接保存了 big_phase_frac
        # ----------------------------------------------------

        big_phase_npz = (

            np.asarray(

                data[
                    "big_phase_frac"
                ],

                dtype=float
            )

            if
            "big_phase_frac"
            in data

            else None
        )

        cycle_no_npz = (

            np.asarray(

                data[
                    "cycle_no"
                ],

                dtype=int
            )

            if
            "cycle_no"
            in data

            else None
        )

    # ========================================================
    # 数据检查
    # ========================================================

    if (
        modes_w.ndim != 3
        or
        modes_w.shape[0]
        <
        N_MODES
    ):

        raise ValueError(

            f"{npz_path}\n"
            f"modes_w shape="
            f"{modes_w.shape}"
        )

    if (
        coeffs.ndim != 2
        or
        coeffs.shape[0]
        <
        N_MODES
    ):

        raise ValueError(

            f"{npz_path}\n"
            f"coeffs shape="
            f"{coeffs.shape}"
        )

    if energy.size < N_MODES:

        raise ValueError(

            f"{npz_path} "
            f"energy 少于 "
            f"{N_MODES} 阶"
        )

    # ========================================================
    # Labels
    # ========================================================

    labels = load_snapshot_labels(

        case,

        coeffs.shape[1]
    )

    # ========================================================
    # Big-period phase
    #
    # 优先级：
    #
    # 1. NPZ
    # 2. snapshot_labels.csv
    # 3. 根据snapshot数自动生成
    # ========================================================

    if (
        big_phase_npz
        is not None
        and
        big_phase_npz.size
        ==
        coeffs.shape[1]
    ):

        big_phase = (
            big_phase_npz
            .copy()
        )

    elif labels:

        big_phase = np.array([

            item[
                "big_phase_frac"
            ]

            for item in labels

        ], dtype=float)

    else:

        print(

            f"警告："
            f"{npz_path}\n"
            f"没有 big_phase_frac，"
            f"按等间距自动生成。"
        )

        big_phase = np.linspace(

            0.0,

            1.0,

            coeffs.shape[1],

            endpoint=False
        )

    # ========================================================
    # Cycle information
    # ========================================================

    if (
        cycle_no_npz
        is not None
        and
        cycle_no_npz.size
        ==
        coeffs.shape[1]
    ):

        cycle_no = (
            cycle_no_npz
            .copy()
        )

    elif labels:

        cycle_no = np.array([

            item[
                "cycle_no"
            ]

            for item in labels

        ], dtype=int)

    else:

        cycle_no = None

    # ========================================================
    # Sort by big-period phase
    # ========================================================

    order = np.argsort(
        big_phase
    )

    big_phase = (
        big_phase[
            order
        ]
    )

    coeffs = (
        coeffs[
            :,
            order
        ]
    )

    if cycle_no is not None:

        cycle_no = (
            cycle_no[
                order
            ]
        )

    return {

        "case":
            case,

        "name":
            case_name_from_values(

                case["k"],

                case["st"],

                case["a"],

                case["dc"]
            ),

        "X":
            X,

        "Y":
            Y,

        "modes_w":
            modes_w[
                :N_MODES
            ].copy(),

        "energy":
            energy[
                :N_MODES
            ].copy(),

        "eigvals":
            (
                None

                if eigvals is None

                else
                eigvals[
                    :N_MODES
                ].copy()
            ),

        "coeffs":
            coeffs[
                :N_MODES
            ].copy(),

        "big_phase":
            big_phase,

        "cycle_no":
            cycle_no,

        "snapshot_labels":
            labels,

        "fluid_mask":
            fluid_mask,

        "npz_path":
            npz_path,
    }


# ============================================================
# 6. 模态涡量加权
# ============================================================

def apply_modal_vorticity_weighting(

    modes_w,

    energy,

    eigvals=None

):

    mode = (
        MODAL_VORTICITY_WEIGHT
        .lower()
    )

    # ========================================================
    # none
    # ========================================================

    if mode == "none":

        weights = np.ones(

            modes_w.shape[0],

            dtype=float
        )

        label = (
            "unweighted"
        )

    # ========================================================
    # sqrt energy
    # ========================================================

    elif mode == "sqrt_energy":

        weights = np.sqrt(

            np.asarray(

                energy[
                    :modes_w.shape[0]
                ],

                dtype=float
            )
        )

        label = (
            r"$\sqrt{E_i}\omega_i$"
        )

    # ========================================================
    # sqrt lambda
    # ========================================================

    elif mode == "sqrt_lambda":

        if eigvals is not None:

            weights = np.sqrt(

                np.asarray(

                    eigvals[
                        :modes_w.shape[0]
                    ],

                    dtype=float
                )
            )

            label = (
                r"$\sqrt{\lambda_i}\omega_i$"
            )

        elif FALLBACK_TO_SQRT_ENERGY:

            weights = np.sqrt(

                np.asarray(

                    energy[
                        :modes_w.shape[0]
                    ],

                    dtype=float
                )
            )

            label = (
                r"$\sqrt{E_i}\omega_i$"
            )

            print(
                "警告：没有 eigvals，"
                "使用 sqrt(E_i)。"
            )

        else:

            raise KeyError(
                "sqrt_lambda 需要 eigvals"
            )

    else:

        raise ValueError(
            "错误的模态加权方式。"
        )

    weights = np.asarray(
        weights,
        dtype=float
    )

    weights[
        ~np.isfinite(
            weights
        )
    ] = 0.0

    fields = (

        modes_w

        *
        weights[
            :,
            None,
            None
        ]
    )

    return (
        fields,
        weights,
        label
    )


# ============================================================
# 7. Display crop
# ============================================================

def crop_by_xlim_ylim(

    X,

    Y,

    fields,

    fluid_mask,

    xlim,

    ylim

):

    display_mask = (

        (X >= xlim[0])

        &

        (X <= xlim[1])

        &

        (Y >= ylim[0])

        &

        (Y <= ylim[1])
    )

    rows = np.where(

        np.any(
            display_mask,
            axis=1
        )

    )[0]

    cols = np.where(

        np.any(
            display_mask,
            axis=0
        )

    )[0]

    if (
        rows.size == 0
        or
        cols.size == 0
    ):

        raise ValueError(
            "显示范围与POD网格无交集。"
        )

    Xc = X[
        np.ix_(
            rows,
            cols
        )
    ]

    Yc = Y[
        np.ix_(
            rows,
            cols
        )
    ]

    Fc = (
        fields[
            :,
            rows,
            :
        ][
            :,
            :,
            cols
        ]
    )

    Mc = fluid_mask[
        np.ix_(
            rows,
            cols
        )
    ]

    return (
        Xc,
        Yc,
        Fc,
        Mc
    )


# ============================================================
# 8. 模态归一化
# ============================================================

def normalise_fields(

    case_results,

    mode="global"

):

    if mode not in [

        "global",

        "row",

        "mode"

    ]:

        raise ValueError(
            "NORMALISE_BY设置错误。"
        )

    eps = 1e-12

    # ========================================================
    # global
    #
    # 全部16工况一起归一化
    # ========================================================

    if mode == "global":

        values = []

        for res in case_results:

            finite = np.abs(

                res[
                    "fields"
                ][
                    np.isfinite(
                        res[
                            "fields"
                        ]
                    )
                ]
            )

            if finite.size:

                values.append(
                    finite
                )

        if not values:

            raise ValueError(
                "没有有限模态涡量值。"
            )

        all_values = np.concatenate(
            values
        )

        scale = max(

            float(
                np.nanpercentile(

                    all_values,

                    NORMALISE_PERCENTILE
                )
            ),

            eps
        )

        print(
            "\nGlobal modal-vorticity "
            f"normalisation scale = "
            f"{scale:.6e}"
        )

        for res in case_results:

            res[
                "fields_plot"
            ] = np.clip(

                res[
                    "fields"
                ]
                /
                scale,

                -1.0,

                1.0
            )

            res[
                "normalise_scale"
            ] = scale

    # ========================================================
    # row
    # ========================================================

    elif mode == "row":

        for res in case_results:

            values = np.abs(

                res[
                    "fields"
                ][
                    np.isfinite(
                        res[
                            "fields"
                        ]
                    )
                ]
            )

            scale = max(

                float(
                    np.nanpercentile(

                        values,

                        NORMALISE_PERCENTILE
                    )
                ),

                eps
            )

            res[
                "fields_plot"
            ] = np.clip(

                res[
                    "fields"
                ]
                /
                scale,

                -1.0,

                1.0
            )

            res[
                "normalise_scale"
            ] = scale

    # ========================================================
    # mode
    # ========================================================

    else:

        for res in case_results:

            fields_plot = np.empty_like(

                res[
                    "fields"
                ]
            )

            scales = []

            for m in range(
                N_MODES
            ):

                values = np.abs(

                    res[
                        "fields"
                    ][m][

                        np.isfinite(

                            res[
                                "fields"
                            ][m]
                        )
                    ]
                )

                scale = max(

                    float(
                        np.nanpercentile(

                            values,

                            NORMALISE_PERCENTILE
                        )
                    ),

                    eps
                )

                fields_plot[m] = np.clip(

                    res[
                        "fields"
                    ][m]
                    /
                    scale,

                    -1.0,

                    1.0
                )

                scales.append(
                    scale
                )

            res[
                "fields_plot"
            ] = fields_plot

            res[
                "normalise_scale"
            ] = scales

    return case_results


# ============================================================
# 9. 大周期 coefficient
# ============================================================

def prepare_big_period_coefficients(
    res: dict
):
    """
    直接使用全部4T快照。

    不再进行单周期截取。
    """

    phase = np.asarray(

        res[
            "big_phase"
        ],

        dtype=float
    )

    coeffs = np.asarray(

        res[
            "coeffs"
        ],

        dtype=float
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    if (
        phase.size
        !=
        coeffs.shape[1]
    ):

        raise ValueError(

            f"{res['name']} "
            f"big_phase与coeffs数量不一致"
        )

    if phase.size < 2:

        raise ValueError(

            f"{res['name']} "
            f"快照数量不足"
        )

    # --------------------------------------------------------
    # phase 修正
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            phase
        )
    ):

        phase = np.linspace(

            0.0,

            1.0,

            coeffs.shape[1],

            endpoint=False
        )

    else:

        order = np.argsort(
            phase
        )

        phase = (
            phase[
                order
            ]
        )

        coeffs = (
            coeffs[
                :,
                order
            ]
        )

    # --------------------------------------------------------
    # 保证位于0~1
    # --------------------------------------------------------

    phase = np.clip(

        phase,

        0.0,

        1.0
    )

    # --------------------------------------------------------
    # 闭合4T大周期
    #
    # 128个快照：
    # 0/128 ... 127/128
    #
    # 在1.0处补回第一个点
    # --------------------------------------------------------

    if phase[-1] < 1.0 - 1e-10:

        phase_plot = np.r_[

            phase,

            1.0
        ]

        coeffs_plot = np.concatenate(

            [

                coeffs,

                coeffs[
                    :,
                    :1
                ]
            ],

            axis=1
        )

    else:

        phase_plot = phase

        coeffs_plot = coeffs

    return (

        phase,

        coeffs,

        phase_plot,

        coeffs_plot
    )


# ============================================================
# 10. coefficient normalisation
# ============================================================

def prepare_coefficients(

    coeffs: np.ndarray,

    normalise: bool

):

    coeffs = np.asarray(

        coeffs,

        dtype=float

    ).copy()

    if not normalise:

        return coeffs

    for i in range(
        coeffs.shape[0]
    ):

        scale = np.nanmax(

            np.abs(
                coeffs[i]
            )
        )

        if (
            np.isfinite(
                scale
            )
            and
            scale > 0.0
        ):

            coeffs[i] /= (
                scale
            )

    return coeffs


# ============================================================
# 11. Layout
# ============================================================

def compute_manual_layout(

    ncases: int,

    ncols: int = 4

):

    total_col_gap = (

        COLUMN_GAP_INCH

        *
        (ncols - 1)
    )

    usable_width = (

        FIG_WIDTH

        -
        MARGIN_LEFT_INCH

        -
        MARGIN_RIGHT_INCH

        -
        total_col_gap
    )

    if usable_width <= 0:

        raise ValueError(
            "Figure width设置错误。"
        )

    axis_width = (

        usable_width

        /
        ncols
    )

    flow_aspect = (

        (Y_LIM[1] - Y_LIM[0])

        /

        (X_LIM[1] - X_LIM[0])
    )

    top_height = (

        axis_width

        *
        flow_aspect
    )

    bottom_height = (

        axis_width

        *
        BOTTOM_ROW_HEIGHT_SCALE
    )

    block_height = (

        top_height

        +
        TOP_BOTTOM_ROW_GAP_INCH

        +
        bottom_height
    )

    coeff_width = (
        axis_width
    )

    phase_width = (
        bottom_height
    )

    bottom_group_width = (

        coeff_width

        +
        3.0
        *
        phase_width

        +
        3.0
        *
        BOTTOM_ROW_EQUAL_GAP_INCH
    )

    available_bottom_width = (

        FIG_WIDTH

        -
        MARGIN_LEFT_INCH

        -
        MARGIN_RIGHT_INCH
    )

    if (
        bottom_group_width
        >
        available_bottom_width
    ):

        raise ValueError(

            "第二行宽度超过图幅。"
        )

    bottom_group_left = (

        MARGIN_LEFT_INCH

        +
        0.5
        *
        (
            available_bottom_width
            -
            bottom_group_width
        )

        +
        BOTTOM_ROW_CENTER_SHIFT_INCH
    )

    bottom_group_right = (

        bottom_group_left

        +
        bottom_group_width
    )

    if (

        bottom_group_left < 0

        or

        bottom_group_right
        >
        FIG_WIDTH

    ):

        raise ValueError(

            "第二行移出图幅。"
        )

    fig_height = (

        MARGIN_TOP_INCH

        +
        MARGIN_BOTTOM_INCH

        +
        ncases
        *
        block_height

        +
        (ncases - 1)
        *
        CASE_BLOCK_GAP_INCH
    )

    top_positions = []

    bottom_positions = []

    block_tops = []

    for i in range(
        ncases
    ):

        block_top = (

            fig_height

            -
            MARGIN_TOP_INCH

            -
            i
            *
            (
                block_height
                +
                CASE_BLOCK_GAP_INCH
            )
        )

        top_bottom = (

            block_top

            -
            top_height
        )

        bottom_bottom = (

            top_bottom

            -
            TOP_BOTTOM_ROW_GAP_INCH

            -
            bottom_height
        )

        block_tops.append(
            block_top
        )

        # ====================================================
        # Top row
        # ====================================================

        row_top = []

        for j in range(
            ncols
        ):

            left = (

                MARGIN_LEFT_INCH

                +
                j
                *
                (
                    axis_width
                    +
                    COLUMN_GAP_INCH
                )
            )

            row_top.append([

                left
                /
                FIG_WIDTH,

                top_bottom
                /
                fig_height,

                axis_width
                /
                FIG_WIDTH,

                top_height
                /
                fig_height
            ])

        # ====================================================
        # Bottom row
        # ====================================================

        row_bottom = []

        current_left = (
            bottom_group_left
        )

        # coefficient
        row_bottom.append([

            current_left
            /
            FIG_WIDTH,

            bottom_bottom
            /
            fig_height,

            coeff_width
            /
            FIG_WIDTH,

            bottom_height
            /
            fig_height
        ])

        current_left += (

            coeff_width

            +
            BOTTOM_ROW_EQUAL_GAP_INCH
        )

        # 3 phase portraits
        for _ in range(
            3
        ):

            row_bottom.append([

                current_left
                /
                FIG_WIDTH,

                bottom_bottom
                /
                fig_height,

                phase_width
                /
                FIG_WIDTH,

                bottom_height
                /
                fig_height
            ])

            current_left += (

                phase_width

                +
                BOTTOM_ROW_EQUAL_GAP_INCH
            )

        top_positions.append(
            row_top
        )

        bottom_positions.append(
            row_bottom
        )

    return (

        fig_height,

        top_positions,

        bottom_positions,

        block_tops,

        axis_width,

        top_height,

        bottom_height
    )


# ============================================================
# 12. Create axes
# ============================================================

def create_axes(

    fig,

    top_positions,

    bottom_positions

):

    ncases = len(
        top_positions
    )

    top_axes = np.empty(

        (
            ncases,
            4
        ),

        dtype=object
    )

    bottom_axes = np.empty(

        (
            ncases,
            4
        ),

        dtype=object
    )

    for i in range(
        ncases
    ):

        for j in range(
            4
        ):

            top_axes[
                i,
                j
            ] = fig.add_axes(

                top_positions[
                    i
                ][j],

                facecolor="none"
            )

            bottom_axes[
                i,
                j
            ] = fig.add_axes(

                bottom_positions[
                    i
                ][j],

                facecolor="none"
            )

            top_axes[
                i,
                j
            ].patch.set_alpha(
                0.0
            )

            bottom_axes[
                i,
                j
            ].patch.set_alpha(
                0.0
            )

    return (
        top_axes,
        bottom_axes
    )


# ============================================================
# 13. Axes style
# ============================================================

def style_axes(
    ax
):

    ax.tick_params(

        direction="in",

        top=True,

        right=True,

        length=2.5,

        width=0.6,

        pad=1.5
    )


# ============================================================
# 14. Plot mode
# ============================================================

def plot_mode(

    ax,

    res: dict,

    mode_index: int,

    pcm_norm

):

    X = res[
        "Xc"
    ]

    Y = res[
        "Yc"
    ]

    field = res[
        "fields_plot"
    ][mode_index]

    pcm = ax.pcolormesh(

        X,
        Y,

        field,

        cmap=
            CMAP,

        norm=
            pcm_norm,

        shading=
            "auto",

        rasterized=
            True
    )

    if SHOW_MASK_CONTOUR:

        ax.contour(

            X,
            Y,

            res[
                "mask"
            ].astype(
                float
            ),

            levels=[
                0.5
            ],

            colors=
                "0.2",

            linewidths=
                0.4
        )

    ax.set_xlim(
        X_LIM
    )

    ax.set_ylim(
        Y_LIM
    )

    ax.set_aspect(

        "equal",

        adjustable=
            "box"
    )

    ax.set_xticks(
        X_TICKS
    )

    ax.set_yticks(
        Y_TICKS
    )

    ax.set_xlabel(

        r"$x/c$",

        labelpad=
            AXIS_LABEL_PAD
    )

    style_axes(
        ax
    )

    if mode_index == 0:

        ax.set_ylabel(

            r"$y/c$",

            labelpad=
                AXIS_LABEL_PAD
        )

    else:

        ax.set_ylabel(
            ""
        )

        ax.tick_params(

            axis=
                "y",

            labelleft=
                False
        )

    # ========================================================
    # Energy
    # ========================================================

    ax.text(

        0.03,
        0.90,

        (
            rf"$E_{mode_index+1}="
            f"{res['energy'][mode_index]*100:.1f}"
            r"\%$"
        ),

        transform=
            ax.transAxes,

        ha=
            "left",

        va=
            "top",

        fontsize=
            ENERGY_LABEL_FONT_SIZE,

        zorder=
            10,

        bbox=dict(
            facecolor="none",
            edgecolor="none",
            alpha=0.0,
            pad=0.8
        )
    )

    return pcm


# ============================================================
# 15. Plot big-period coefficients
# ============================================================

def plot_big_period_coefficients(

    ax,

    res: dict,

    global_ylim,

    show_legend=False

):

    phase_plot = res[
        "phase_plot"
    ]

    coeffs_plot = (
        prepare_coefficients(

            res[
                "coeffs_big_plot"
            ],

            NORMALISE_TIME_COEFFS
        )
    )

    # ========================================================
    # a1-a4
    # ========================================================

    for m in range(
        N_MODES
    ):

        kwargs = {}

        if COEFF_SHOW_MARKERS:

            kwargs.update(

                marker=
                    "o",

                ms=
                    COEFF_MARKER_SIZE,

                markevery=
                    max(
                        1,
                        len(
                            phase_plot
                        )
                        //
                        16
                    )
            )

        ax.plot(

            phase_plot,

            coeffs_plot[m],

            lw=
                COEFF_LINE_WIDTH,

            label=
                rf"$a_{m+1}$",

            **kwargs
        )

    # ========================================================
    # y=0
    # ========================================================

    ax.axhline(

        0.0,

        color=
            "0.65",

        lw=
            0.45,

        ls=
            "--",

        zorder=
            0
    )

    # ========================================================
    # 四个普通周期的边界
    # ========================================================

    if SHOW_SUBCYCLE_DIVIDERS:

        for x in SUBCYCLE_DIVIDERS:

            ax.axvline(

                x,

                color=
                    "0.72",

                lw=
                    0.5,

                ls=
                    "--",

                zorder=
                    0
            )

    # ========================================================
    # Axis
    # ========================================================

    ax.set_xlim(
        0.0,
        1.0
    )

    ax.set_xticks(

        COEFF_X_TICKS,

        labels=
            COEFF_X_TICK_LABELS
    )

    ax.set_yticks(
        COEFF_Y_TICKS
    )

    if global_ylim is not None:

        ax.set_ylim(
            global_ylim
        )

    # --------------------------------------------------------
    # 注意这里已经不是 t/T
    # --------------------------------------------------------

    ax.set_xlabel(

        r"$\tau/(4T)$",

        labelpad=
            AXIS_LABEL_PAD
    )

    if NORMALISE_TIME_COEFFS:

        ax.set_ylabel(

            r"$a_i/\max|a_i|$",

            labelpad=
                COEFF_Y_LABEL_PAD
        )

    else:

        ax.set_ylabel(

            r"$a_i$",

            rotation=
                0,

            ha=
                "right",

            va=
                "center",

            labelpad=
                COEFF_Y_LABEL_PAD
        )

    style_axes(
        ax
    )

    if show_legend:

        ax.legend(

            frameon=
                False,

            ncol=
                COEFF_LEGEND_NCOL,

            loc=
                COEFF_LEGEND_LOC,

            bbox_to_anchor=
                COEFF_LEGEND_BBOX,

            handlelength=
                1.4,

            columnspacing=
                0.35,

            handletextpad=
                0.25,

            borderaxespad=
                0.15
        )


# ============================================================
# 16. Phase portrait
# ============================================================

def plot_phase_portrait(

    ax,

    res: dict,

    x_mode: int,

    y_mode: int,

    show_ylabel=True

):

    # --------------------------------------------------------
    # 使用整个 4T 大周期
    # --------------------------------------------------------

    coeffs = prepare_coefficients(

        res[
            "coeffs_big"
        ],

        NORMALISE_PHASE_COEFFS
    )

    x = coeffs[
        x_mode
    ]

    y = coeffs[
        y_mode
    ]

    if (
        CLOSE_PHASE_PORTRAIT
        and
        x.size > 1
    ):

        x_line = np.r_[

            x,

            x[0]
        ]

        y_line = np.r_[

            y,

            y[0]
        ]

    else:

        x_line = x
        y_line = y

    # ========================================================
    # phase trajectory
    # ========================================================

    ax.plot(

        x_line,

        y_line,

        "-",

        lw=
            PHASE_LINE_WIDTH,

        alpha=
            0.9
    )

    # ========================================================
    # Zero lines
    # ========================================================

    if PHASE_SHOW_ZERO_LINES:

        ax.axhline(

            0.0,

            color=
                "0.65",

            lw=
                0.45,

            ls=
                "--",

            zorder=
                0
        )

        ax.axvline(

            0.0,

            color=
                "0.65",

            lw=
                0.45,

            ls=
                "--",

            zorder=
                0
        )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(

        PHASE_AXIS_MIN,

        PHASE_AXIS_MAX
    )

    ax.set_ylim(

        PHASE_AXIS_MIN,

        PHASE_AXIS_MAX
    )

    ax.set_xticks(
        PHASE_TICKS
    )

    ax.set_yticks(
        PHASE_TICKS
    )

    ax.set_aspect(

        "equal",

        adjustable=
            "box"
    )

    ax.set_box_aspect(
        1.0
    )

    style_axes(
        ax
    )

    if NORMALISE_PHASE_COEFFS:

        xlabel = (

            rf"$a_{x_mode+1}/"
            rf"\max|a_{x_mode+1}|$"
        )

        ylabel = (

            rf"$a_{y_mode+1}/"
            rf"\max|a_{y_mode+1}|$"
        )

    else:

        xlabel = (
            rf"$a_{x_mode+1}$"
        )

        ylabel = (
            rf"$a_{y_mode+1}$"
        )

    ax.set_xlabel(

        xlabel,

        labelpad=
            PHASE_AXIS_LABEL_PAD
    )

    if show_ylabel:

        ax.set_ylabel(

            ylabel,

            rotation=
                0,

            ha=
                "right",

            va=
                "center",

            labelpad=
                PHASE_Y_LABEL_PAD
        )

    else:

        ax.set_ylabel(
            ""
        )

        ax.tick_params(

            axis=
                "y",

            labelleft=
                False
        )


# ============================================================
# 17. Load all 16 cases
# ============================================================

def load_and_prepare_all_cases():

    case_results = []

    print(
        "=" * 72
    )

    print(
        "读取新的 DC 大周期 POD 结果"
    )

    print(
        "=" * 72
    )

    for case in ALL_CASES:

        print(

            f"Loading: "
            f"k={case['k']}, "
            f"St={case['st']}, "
            f"a={case['a']}, "
            f"DC={case['dc']}"
        )

        try:

            res = load_case_pod_result(
                case
            )

        except Exception as e:

            print(
                f"  [跳过] {repr(e)}"
            )

            continue

        # ====================================================
        # Modal weighting
        # ====================================================

        (
            weighted_modes_w,

            modal_weights,

            modal_field_label

        ) = apply_modal_vorticity_weighting(

            res[
                "modes_w"
            ],

            res[
                "energy"
            ],

            res.get(
                "eigvals"
            )
        )

        # ====================================================
        # Crop
        # ====================================================

        (
            Xc,

            Yc,

            Fc,

            Mc

        ) = crop_by_xlim_ylim(

            res[
                "X"
            ],

            res[
                "Y"
            ],

            weighted_modes_w,

            res[
                "fluid_mask"
            ],

            X_LIM,

            Y_LIM
        )

        # ====================================================
        # Mask
        # ====================================================

        for m in range(
            N_MODES
        ):

            Fc[m][
                ~Mc
            ] = np.nan

        # ====================================================
        # 直接使用整个4T
        # ====================================================

        (
            phase,

            coeffs_big,

            phase_plot,

            coeffs_big_plot

        ) = prepare_big_period_coefficients(
            res
        )

        res[
            "Xc"
        ] = Xc

        res[
            "Yc"
        ] = Yc

        res[
            "fields"
        ] = Fc

        res[
            "mask"
        ] = Mc

        res[
            "modal_weights"
        ] = modal_weights

        res[
            "modal_field_label"
        ] = modal_field_label

        res[
            "phase"
        ] = phase

        res[
            "coeffs_big"
        ] = coeffs_big

        res[
            "phase_plot"
        ] = phase_plot

        res[
            "coeffs_big_plot"
        ] = coeffs_big_plot

        case_results.append(
            res
        )

    if not case_results:

        raise RuntimeError(
            "没有成功读取任何POD结果。"
        )

    # ========================================================
    # 重要：
    #
    # 在这里对全部16工况统一归一化。
    #
    # 而不是每个 DC 单独归一化。
    # ========================================================

    case_results = normalise_fields(

        case_results,

        NORMALISE_BY
    )

    return case_results


# ============================================================
# 18. Get one DC
# ============================================================

def get_results_for_dc(

    all_results,

    dc

):

    result_map = {}

    for res in all_results:

        if np.isclose(

            res[
                "case"
            ][
                "dc"
            ],

            dc
        ):

            result_map[

                res[
                    "case"
                ][
                    "k"
                ]

            ] = res

    ordered = []

    for k in K_LIST:

        if k in result_map:

            ordered.append(
                result_map[k]
            )

    return ordered


# ============================================================
# 19. Plot one DC figure
# ============================================================

def plot_one_dc_figure(

    case_results,

    dc

):

    if not case_results:

        print(
            f"DC={dc} 没有数据，跳过。"
        )

        return

    # ========================================================
    # Coefficient global ylim
    # ========================================================

    if COEFF_Y_LIM is not None:

        coeff_global_ylim = tuple(
            COEFF_Y_LIM
        )

    else:

        values = []

        for res in case_results:

            coeffs_plot = (
                prepare_coefficients(

                    res[
                        "coeffs_big_plot"
                    ],

                    NORMALISE_TIME_COEFFS
                )
            )

            finite = np.abs(

                coeffs_plot[

                    np.isfinite(
                        coeffs_plot
                    )
                ]
            )

            if finite.size:

                values.append(
                    finite
                )

        if values:

            vmax = float(

                np.nanpercentile(

                    np.concatenate(
                        values
                    ),

                    99.5
                )
            )

            vmax = max(

                vmax * 1.08,

                1e-12
            )

            coeff_global_ylim = (
                -vmax,
                vmax
            )

        else:

            coeff_global_ylim = None

    # ========================================================
    # Layout
    # ========================================================

    ncases = len(
        case_results
    )

    (
        fig_height,

        top_positions,

        bottom_positions,

        block_tops,

        axis_width,

        top_height,

        bottom_height

    ) = compute_manual_layout(
        ncases
    )

    print(
        "\n"
        f"DC={dc:g}: "
        f"Figure size = "
        f"{FIG_WIDTH:.3f} × "
        f"{fig_height:.3f} in"
    )

    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(

        figsize=(
            FIG_WIDTH,
            fig_height
        ),

        facecolor=
            "none"
    )

    fig.patch.set_alpha(
        0.0
    )

    (
        top_axes,

        bottom_axes

    ) = create_axes(

        fig,

        top_positions,

        bottom_positions
    )

    pcm_norm = mcolors.TwoSlopeNorm(

        vmin=
            -1.0,

        vcenter=
            0.0,

        vmax=
            1.0
    )

    pcm_for_colorbar = None

    # ========================================================
    # Draw
    # ========================================================

    for i, res in enumerate(
        case_results
    ):

        # ====================================================
        # 第一行：
        # Mode 1 - Mode 4
        # ====================================================

        for j in range(
            N_MODES
        ):

            pcm = plot_mode(

                top_axes[
                    i,
                    j
                ],

                res,

                j,

                pcm_norm
            )

            if (
                i == 0
                and
                j == 3
            ):

                pcm_for_colorbar = (
                    pcm
                )

            add_inner_panel_label(

                top_axes[
                    i,
                    j
                ],

                j
            )

        # ====================================================
        # 第二行：
        # 整个4T的 a1-a4
        # ====================================================

        plot_big_period_coefficients(

            bottom_axes[
                i,
                0
            ],

            res,

            coeff_global_ylim,

            show_legend=
                (
                    i == 0
                )
        )

        add_inner_panel_label(

            bottom_axes[
                i,
                0
            ],

            4
        )

        # ====================================================
        # a1-a2
        # ====================================================

        plot_phase_portrait(

            bottom_axes[
                i,
                1
            ],

            res,

            x_mode=
                0,

            y_mode=
                1,

            show_ylabel=
                True
        )

        add_inner_panel_label(

            bottom_axes[
                i,
                1
            ],

            5
        )

        # ====================================================
        # a3-a4
        # ====================================================

        plot_phase_portrait(

            bottom_axes[
                i,
                2
            ],

            res,

            x_mode=
                2,

            y_mode=
                3,

            show_ylabel=
                True
        )

        add_inner_panel_label(

            bottom_axes[
                i,
                2
            ],

            6
        )

        # ====================================================
        # a1-a3
        # ====================================================

        plot_phase_portrait(

            bottom_axes[
                i,
                3
            ],

            res,

            x_mode=
                0,

            y_mode=
                2,

            show_ylabel=
                True
        )

        add_inner_panel_label(

            bottom_axes[
                i,
                3
            ],

            7
        )


    # ========================================================
    # Shared colorbar
    # ========================================================

    if (
        SHOW_INSET_COLORBAR
        and
        pcm_for_colorbar
        is not None
    ):

        target_ax = top_axes[
            0,
            3
        ]

        cax = target_ax.inset_axes(

            INSET_COLORBAR_BOUNDS,

            facecolor=
                "none"
        )

        cax.patch.set_alpha(
            0.0
        )

        cbar = fig.colorbar(

            pcm_for_colorbar,

            cax=
                cax,

            orientation=
                "horizontal",

            ticks=
                COLORBAR_TICKS
        )

        cbar.ax.tick_params(

            direction=
                "in",

            length=
                1.8,

            width=
                0.5,

            labelsize=
                COLORBAR_TICK_LABEL_SIZE,

            pad=
                0.8
        )


    # ========================================================
    # 主面板标签 (a)-(d)
    # ========================================================

    fig.canvas.draw()

    renderer = (
        fig.canvas
        .get_renderer()
    )

    for i in range(
        ncases
    ):

        panel_axes = (

            [
                top_axes[
                    i,
                    j
                ]
                for j in range(
                    N_MODES
                )
            ]

            +

            [
                bottom_axes[
                    i,
                    j
                ]
                for j in range(
                    4
                )
            ]
        )

        tight_bboxes = [

            ax.get_tightbbox(
                renderer
            )

            for ax in panel_axes

            if ax.get_visible()
        ]

        tight_bboxes = [

            bbox

            for bbox in tight_bboxes

            if bbox is not None
        ]

        if not tight_bboxes:

            continue

        panel_bbox_display = (
            Bbox.union(
                tight_bboxes
            )
        )

        panel_bbox_figure = (
            panel_bbox_display
            .transformed(
                fig.transFigure
                .inverted()
            )
        )

        panel_letter = chr(

            ord("a")

            +
            i
        )

        fig.text(

            panel_bbox_figure.x0,

            panel_bbox_figure.y1,

            rf"$({panel_letter})$",

            transform=
                fig.transFigure,

            ha=
                "center",

            va=
                "center",

            fontsize=
                MAIN_PANEL_LABEL_FONT_SIZE
        )


    # ========================================================
    # Output filename
    # ========================================================

    dc_string = (
        fmt_num_short(
            dc
        )
    )

    basename = (

        OUT_PREFIX

        +
        dc_string
    )

    png_path = os.path.join(

        FIG_SAVE_DIR,

        basename
        +
        ".png"
    )

    svg_path = os.path.join(

        FIG_SAVE_DIR,

        basename
        +
        ".svg"
    )

    pdf_path = os.path.join(

        FIG_SAVE_DIR,

        basename
        +
        ".pdf"
    )


    # ========================================================
    # Save
    # ========================================================

    fig.savefig(

        png_path,

        dpi=
            DPI,

        transparent=
            TRANSPARENT_SVG_BACKGROUND,

        facecolor=
            (
                "none"
                if
                TRANSPARENT_SVG_BACKGROUND
                else
                "white"
            ),

        edgecolor=
            "none"
    )

    fig.savefig(

        svg_path,

        dpi=
            DPI,

        transparent=
            TRANSPARENT_SVG_BACKGROUND,

        facecolor=
            (
                "none"
                if
                TRANSPARENT_SVG_BACKGROUND
                else
                "white"
            ),

        edgecolor=
            "none"
    )

    # 如需要 PDF：
    #
    # fig.savefig(
    #     pdf_path,
    #     dpi=DPI,
    #     transparent=True
    # )

    plt.close(
        fig
    )

    # ========================================================
    # Print information
    # ========================================================

    print(
        "\n"
        f"DC={dc:g} 图已保存："
    )

    print(
        png_path
    )

    print(
        svg_path
    )

    print(
        "\n读取的 POD 数据："
    )

    for res in case_results:

        case = res[
            "case"
        ]

        print(

            f"  k={case['k']:g}, "
            f"a={case['a']:g}, "
            f"DC={case['dc']:g}"
        )

        print(
            f"    {res['npz_path']}"
        )

        print(

            f"    snapshots = "
            f"{res['coeffs_big'].shape[1]}"
        )

        print(

            f"    big phase = "
            f"{res['phase'][0]:.5f}"
            f" ~ "
            f"{res['phase'][-1]:.5f}"
        )

        if (
            res[
                "cycle_no"
            ]
            is not None
        ):

            cycles = np.unique(

                res[
                    "cycle_no"
                ]
            )

            print(

                "    cycles = "

                +
                ", ".join(
                    str(int(x))
                    for x in cycles
                )
            )

        print(

            "    E1-E4 = "

            +

            ", ".join(

                f"{e:.4f}"

                for e in
                res[
                    "energy"
                ]
            )
        )

        print(

            "    modal weights = "

            +

            ", ".join(

                f"{w:.4e}"

                for w in
                res[
                    "modal_weights"
                ]
            )
        )


# ============================================================
# 20. Main
# ============================================================

def main():

    # ========================================================
    # 先一次性读取全部16个工况
    #
    # 这是为了实现真正的 global normalization。
    # ========================================================

    all_results = (
        load_and_prepare_all_cases()
    )

    print(
        "\n"
        "=" * 72
    )

    print(

        f"成功读取 "
        f"{len(all_results)} "
        f"/16 个POD工况"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 每个DC单独生成一张图
    # ========================================================

    for dc in DC_LIST:

        case_results = (
            get_results_for_dc(

                all_results,

                dc
            )
        )

        plot_one_dc_figure(

            case_results,

            dc
        )

    print(
        "\n"
        "=" * 72
    )

    print(
        "所有 DC POD 论文图绘制完成"
    )

    print(

        f"保存目录："
        f"{FIG_SAVE_DIR}"
    )

    print(
        "=" * 72
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()