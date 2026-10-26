# -*- coding: utf-8 -*-
"""
plot_fig_pod_k1p4_modes_only.py

固定工况：
    St = 0.3
    k = 1.4
    alpha0 = 17 deg

绘制：
    DC = 1.00, 0.75, 0.50, 0.25

整张图包含四个主子图：
    (a) DC = 1.00
    (b) DC = 0.75
    (c) DC = 0.50
    (d) DC = 0.25

每个主子图仅包含一行：
    Mode 1
    Mode 2
    Mode 3
    Mode 4

不再绘制：
    - 时间系数 a1-a4
    - a1-a2 phase portrait
    - a3-a4 phase portrait
    - a1-a3 phase portrait

四种 DC 的所有模态统一进行 global normalization，
因此不同 DC 之间的模态强度仍可直接比较。
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import colormaps
from matplotlib.transforms import Bbox


# ============================================================
# 1. 用户配置区
# ============================================================

# ------------------------------------------------------------
# POD 结果路径
# ------------------------------------------------------------

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
# 固定工况
# ============================================================

ST = 0.3
K = 1.4
ALPHA0 = 17

DC_LIST = [
    1.00,
    0.75,
    0.50,
    0.25,
]


# ============================================================
# 生成四个工况
# ============================================================

ALL_CASES = []

for dc in DC_LIST:

    ALL_CASES.append({
        "k": K,
        "st": ST,
        "a": ALPHA0,
        "dc": dc,
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
# ============================================================

MODAL_VORTICITY_WEIGHT = "sqrt_lambda"

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
# 模态归一化
#
# global:
#   k=1.4 下全部
#
#       4 DC × 4 modes
#
#   共用同一个归一化尺度。
#
# 因此不同 DC 之间能够直接比较模态幅值。
# ============================================================

NORMALISE_BY = "global"

NORMALISE_PERCENTILE = 99.0


# ============================================================
# Colormap
# ============================================================

CMAP_VMIN = -1.0
CMAP_VMAX = 1.0

WHITE_SIGMA = 0.16

SHOW_MASK_CONTOUR = False


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


# ============================================================
# Figure size / layout
# ============================================================

FIG_WIDTH = 6.5

MARGIN_LEFT_INCH = 0.62
MARGIN_RIGHT_INCH = 0.18

MARGIN_TOP_INCH = 0.40
MARGIN_BOTTOM_INCH = 0.40

# Mode 1-4 之间的水平距离
COLUMN_GAP_INCH = 0.085

# DC 主子图之间的竖直间距
CASE_BLOCK_GAP_INCH = 0.14


# ============================================================
# Colorbar
#
# 放在第一行 Mode 4 内部
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

OUT_BASENAME = "Fig_POD_k1p4_modes_only"

DPI = 600


# ============================================================
# Transparent background
# ============================================================

TRANSPARENT_BACKGROUND = True

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
# 3. Colormap
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

    cmap = mcolors.ListedColormap(
        new_colors
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
    sigma=WHITE_SIGMA
)


# ============================================================
# 4. Naming
# ============================================================

def fmt_num(
    x: float
) -> str:
    """
    与 POD 结果命名保持一致。

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

    cname = case_name_from_values(
        case["k"],
        case["st"],
        case["a"],
        case["dc"]
    )

    return os.path.join(
        SAVE_ROOT,
        cname,
        f"{cname}_pod_results.npz"
    )


# ============================================================
# 5. 内部小图标签
# ============================================================

def roman_panel_label(
    index: int
) -> str:

    roman = [
        "i",
        "ii",
        "iii",
        "iv"
    ]

    if not (
        0 <= index < len(roman)
    ):

        raise ValueError(
            "内部编号只支持 0-3。"
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

        transform=ax.transAxes,

        ha="right",
        va="top",

        fontsize=
            INNER_PANEL_LABEL_FONT_SIZE,

        zorder=20,

        bbox=dict(
            facecolor="none",
            edgecolor="none",
            alpha=0.0,
            pad=0.8
        )
    )


# ============================================================
# 6. 读取单个 POD
# ============================================================

def load_case_pod_result(
    case: dict
) -> dict:

    npz_path = get_npz_path(
        case
    )

    if not os.path.exists(
        npz_path
    ):

        raise FileNotFoundError(
            f"未找到 POD 文件：\n"
            f"{npz_path}"
        )

    with np.load(
        npz_path
    ) as data:

        required_keys = [
            "X",
            "Y",
            "modes_w",
            "energy",
            "fluid_mask"
        ]

        for key in required_keys:

            if key not in data:

                raise KeyError(
                    f"{npz_path} 中缺少 {key}"
                )

        X = np.asarray(
            data["X"]
        )

        Y = np.asarray(
            data["Y"]
        )

        modes_w = np.asarray(
            data["modes_w"],
            dtype=float
        )

        energy = np.asarray(
            data["energy"],
            dtype=float
        )

        fluid_mask = np.asarray(
            data["fluid_mask"],
            dtype=bool
        )

        eigvals = (
            np.asarray(
                data["eigvals"],
                dtype=float
            )

            if "eigvals" in data

            else None
        )

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

    if (
        modes_w.ndim != 3
        or
        modes_w.shape[0] < N_MODES
    ):

        raise ValueError(
            f"{npz_path}\n"
            f"modes_w shape="
            f"{modes_w.shape}"
        )

    if energy.size < N_MODES:

        raise ValueError(
            f"{npz_path} "
            f"energy 少于 {N_MODES} 阶"
        )

    if (
        X.shape != Y.shape
        or
        X.shape != fluid_mask.shape
    ):

        raise ValueError(
            f"{npz_path}\n"
            "X、Y、fluid_mask 尺寸不一致。"
        )

    if (
        modes_w.shape[1:] != X.shape
    ):

        raise ValueError(
            f"{npz_path}\n"
            f"modes_w spatial shape="
            f"{modes_w.shape[1:]}, "
            f"grid shape={X.shape}"
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

                else eigvals[
                    :N_MODES
                ].copy()
            ),

        "fluid_mask":
            fluid_mask,

        "npz_path":
            npz_path,
    }


# ============================================================
# 7. 模态涡量加权
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

    # --------------------------------------------------------
    # none
    # --------------------------------------------------------

    if mode == "none":

        weights = np.ones(
            modes_w.shape[0],
            dtype=float
        )

        label = "unweighted"

    # --------------------------------------------------------
    # sqrt energy
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # sqrt lambda
    # --------------------------------------------------------

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
# 8. Display crop
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
            "显示范围与 POD 网格无交集。"
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
# 9. 模态归一化
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
            "NORMALISE_BY 设置错误。"
        )

    eps = 1e-12

    # ========================================================
    # global
    #
    # 四种 DC 的所有 Mode 一起归一化
    # ========================================================

    if mode == "global":

        values = []

        for res in case_results:

            field = res[
                "fields"
            ]

            finite = np.abs(
                field[
                    np.isfinite(
                        field
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
            "normalisation scale = "
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
    #
    # 每个 DC 自己归一化
    # ========================================================

    elif mode == "row":

        for res in case_results:

            field = res[
                "fields"
            ]

            values = np.abs(
                field[
                    np.isfinite(
                        field
                    )
                ]
            )

            if values.size == 0:

                raise ValueError(
                    f"{res['name']} "
                    "没有有限模态涡量值。"
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
                field
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
    #
    # 每一阶单独归一化
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

                field = res[
                    "fields"
                ][m]

                values = np.abs(
                    field[
                        np.isfinite(
                            field
                        )
                    ]
                )

                if values.size == 0:

                    scale = eps

                else:

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
                    field
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
# 10. Layout
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
            "Figure width 设置错误。"
        )

    axis_width = (
        usable_width
        /
        ncols
    )

    # --------------------------------------------------------
    # 保持流场 x-y 等比例
    # --------------------------------------------------------

    flow_aspect = (
        (Y_LIM[1] - Y_LIM[0])
        /
        (X_LIM[1] - X_LIM[0])
    )

    axis_height = (
        axis_width
        *
        flow_aspect
    )

    fig_height = (
        MARGIN_TOP_INCH
        +
        MARGIN_BOTTOM_INCH
        +
        ncases
        *
        axis_height
        +
        (ncases - 1)
        *
        CASE_BLOCK_GAP_INCH
    )

    positions = []

    for i in range(
        ncases
    ):

        row_top = (
            fig_height
            -
            MARGIN_TOP_INCH
            -
            i
            *
            (
                axis_height
                +
                CASE_BLOCK_GAP_INCH
            )
        )

        row_bottom = (
            row_top
            -
            axis_height
        )

        row_positions = []

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

            row_positions.append([
                left
                /
                FIG_WIDTH,

                row_bottom
                /
                fig_height,

                axis_width
                /
                FIG_WIDTH,

                axis_height
                /
                fig_height
            ])

        positions.append(
            row_positions
        )

    return (
        fig_height,
        positions,
        axis_width,
        axis_height
    )


# ============================================================
# 11. Create axes
# ============================================================

def create_axes(
    fig,
    positions
):

    ncases = len(
        positions
    )

    axes = np.empty(
        (
            ncases,
            N_MODES
        ),
        dtype=object
    )

    for i in range(
        ncases
    ):

        for j in range(
            N_MODES
        ):

            axes[
                i,
                j
            ] = fig.add_axes(
                positions[
                    i
                ][j],
                facecolor="none"
            )

            axes[
                i,
                j
            ].patch.set_alpha(
                0.0
            )

    return axes


# ============================================================
# 12. Axes style
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
# 13. Plot mode
# ============================================================

def plot_mode(
    ax,
    res: dict,
    mode_index: int,
    pcm_norm,
    show_xaxis: bool = True
):

    X = res[
        "Xc"
    ]

    Y = res[
        "Yc"
    ]

    field = res[
        "fields_plot"
    ][
        mode_index
    ]

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

    # --------------------------------------------------------
    # Body / fluid mask contour
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlim(
        X_LIM
    )

    ax.set_ylim(
        Y_LIM
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xticks(
        X_TICKS
    )

    ax.set_yticks(
        Y_TICKS
    )

    # ========================================================
    # 仅最下方一行显示横坐标值和标签
    # ========================================================
    if show_xaxis:

        ax.set_xlabel(
            r"$x/c$",
            labelpad=
                AXIS_LABEL_PAD
        )

        ax.tick_params(
            axis="x",
            labelbottom=True
        )

    else:

        ax.set_xlabel(
            ""
        )

        ax.tick_params(
            axis="x",
            labelbottom=False
        )

    style_axes(
        ax
    )

    # --------------------------------------------------------
    # 只有每一行第一张图保留 y 轴
    # --------------------------------------------------------

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
            axis="y",
            labelleft=False
        )

    # ========================================================
    # Energy
    # ========================================================

    ax.text(
        0.03,
        0.90,

        (
            rf"$E_{mode_index + 1}="
            f"{res['energy'][mode_index] * 100:.1f}"
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
# 14. Load and prepare all k=1.4 cases
# ============================================================

def load_and_prepare_all_cases():

    case_results = []

    print(
        "=" * 72
    )

    print(
        "读取 k=1.4 的 DC 大周期 POD 模态结果"
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

        case_results.append(
            res
        )

    if not case_results:

        raise RuntimeError(
            "没有成功读取任何 POD 结果。"
        )

    # ========================================================
    # 按 DC_LIST 顺序重新排列
    # ========================================================

    result_map = {
        float(
            res["case"]["dc"]
        ):
        res

        for res in case_results
    }

    ordered_results = []

    for dc in DC_LIST:

        matched = None

        for key, res in result_map.items():

            if np.isclose(
                key,
                dc
            ):

                matched = res
                break

        if matched is not None:

            ordered_results.append(
                matched
            )

    # ========================================================
    # 四种 DC 统一归一化
    # ========================================================

    ordered_results = normalise_fields(
        ordered_results,
        NORMALISE_BY
    )

    return ordered_results


# ============================================================
# 15. Plot
# ============================================================

def plot_figure(
    case_results
):

    ncases = len(
        case_results
    )

    (
        fig_height,
        positions,
        axis_width,
        axis_height

    ) = compute_manual_layout(
        ncases
    )

    print(
        "\n"
        f"Figure size = "
        f"{FIG_WIDTH:.3f} × "
        f"{fig_height:.3f} in"
    )

    print(
        f"Single modal axis = "
        f"{axis_width:.3f} × "
        f"{axis_height:.3f} in"
    )

    # ========================================================
    # Figure
    # ========================================================

    fig = plt.figure(
        figsize=(
            FIG_WIDTH,
            fig_height
        ),
        facecolor="none"
    )

    fig.patch.set_alpha(
        0.0
    )

    axes = create_axes(
        fig,
        positions
    )

    pcm_norm = mcolors.TwoSlopeNorm(
        vmin=-1.0,
        vcenter=0.0,
        vmax=1.0
    )

    pcm_for_colorbar = None

    # ========================================================
    # Draw four DC rows
    # ========================================================

    for i, res in enumerate(
        case_results
    ):

        for j in range(
            N_MODES
        ):

            pcm = plot_mode(
                axes[
                    i,
                    j
                ],

                res,

                j,

                pcm_norm,

                show_xaxis=(
                    i == ncases - 1
                )
            )

            # -----------------------------------------------
            # 使用第一行 Mode 4 创建共享色条
            # -----------------------------------------------

            if (
                i == 0
                and
                j == 3
            ):

                pcm_for_colorbar = (
                    pcm
                )

            # -----------------------------------------------
            # (i)-(iv)
            # -----------------------------------------------

            add_inner_panel_label(
                axes[
                    i,
                    j
                ],
                j
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

        target_ax = axes[
            0,
            3
        ]

        cax = target_ax.inset_axes(
            INSET_COLORBAR_BOUNDS,
            facecolor="none"
        )

        cax.patch.set_alpha(
            0.0
        )

        cbar = fig.colorbar(
            pcm_for_colorbar,
            cax=cax,
            orientation="horizontal",
            ticks=
                COLORBAR_TICKS
        )

        cbar.ax.tick_params(
            direction="in",
            length=1.8,
            width=0.5,
            labelsize=
                COLORBAR_TICK_LABEL_SIZE,
            pad=0.8
        )

    # ========================================================
    # 主面板标签 (a)-(d)
    #
    # 每一行对应一个 DC
    # ========================================================

    fig.canvas.draw()

    renderer = (
        fig.canvas
        .get_renderer()
    )

    for i in range(
        ncases
    ):

        row_axes = [
            axes[
                i,
                j
            ]

            for j in range(
                N_MODES
            )
        ]

        tight_bboxes = [
            ax.get_tightbbox(
                renderer
            )

            for ax in row_axes

            if ax.get_visible()
        ]

        tight_bboxes = [
            bbox
            for bbox in tight_bboxes
            if bbox is not None
        ]

        if not tight_bboxes:

            continue

        row_bbox_display = Bbox.union(
            tight_bboxes
        )

        row_bbox_figure = (
            row_bbox_display
            .transformed(
                fig.transFigure.inverted()
            )
        )

        panel_letter = chr(
            ord("a") + i
        )

        # ----------------------------------------------------
        # 放在每一行整体范围左上角外侧
        # ----------------------------------------------------

        fig.text(
            row_bbox_figure.x0 - 0.008,
            row_bbox_figure.y1,

            rf"$({panel_letter})$",

            transform=
                fig.transFigure,

            ha=
                "right",

            va=
                "top",

            fontsize=
                MAIN_PANEL_LABEL_FONT_SIZE
        )

    # ========================================================
    # Output
    # ========================================================

    png_path = os.path.join(
        FIG_SAVE_DIR,
        OUT_BASENAME
        +
        ".png"
    )

    svg_path = os.path.join(
        FIG_SAVE_DIR,
        OUT_BASENAME
        +
        ".svg"
    )

    # ========================================================
    # Save PNG
    # ========================================================

    fig.savefig(
        png_path,

        dpi=
            DPI,

        transparent=
            TRANSPARENT_BACKGROUND,

        facecolor=
            (
                "none"
                if
                TRANSPARENT_BACKGROUND
                else
                "white"
            ),

        edgecolor=
            "none"
    )

    # ========================================================
    # Save SVG
    # ========================================================

    fig.savefig(
        svg_path,

        dpi=
            DPI,

        transparent=
            TRANSPARENT_BACKGROUND,

        facecolor=
            (
                "none"
                if
                TRANSPARENT_BACKGROUND
                else
                "white"
            ),

        edgecolor=
            "none"
    )

    plt.close(
        fig
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n图已保存："
    )

    print(
        png_path
    )

    print(
        svg_path
    )

    print(
        "\n主子图对应关系："
    )

    for i, res in enumerate(
        case_results
    ):

        case = res[
            "case"
        ]

        panel_letter = chr(
            ord("a") + i
        )

        print(
            f"  ({panel_letter}) "
            f"DC={case['dc']:g}"
        )

    print(
        "\n读取的 POD 数据："
    )

    for res in case_results:

        case = res[
            "case"
        ]

        print(
            f"\n  k={case['k']:g}, "
            f"St={case['st']:g}, "
            f"a={case['a']:g}, "
            f"DC={case['dc']:g}"
        )

        print(
            f"    {res['npz_path']}"
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
# 16. Main
# ============================================================

def main():

    all_results = (
        load_and_prepare_all_cases()
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        f"成功读取 "
        f"{len(all_results)} "
        f"/{len(ALL_CASES)} 个 POD 工况"
    )

    print(
        "=" * 72
    )

    plot_figure(
        all_results
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "k=1.4 POD 模态图绘制完成"
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