# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, label


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

    # SVG 中文字保持可编辑
    "svg.fonttype": "none",
})

if USE_TEX:
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "text.latex.preamble":
            r"\usepackage{amsmath} "
            r"\usepackage{newtxtext,newtxmath}"
    })


# ============================================================
# User parameters
# ============================================================
u = 1.0
rou = 1.0
c = 1.0

a = 0
dc = 1


# ============================================================
# Selected cases
# ============================================================
SELECTED_CASES = [
    (0.5, 0.2),
    (0.8, 0.2),
    (1.1, 0.3),
    (1.7, 0.4),

    (2.0, 0.2),
    (0.8, 0.4),
    (0.8, 0.5),
    (0.5, 0.5),
]


# ============================================================
# Cycle / phase
# ============================================================
TARGET_CYCLE = 15
TARGET_PHASE = 0.25
PHASE_TOL = 0.04

# F165000.dat -> t = 16.5000
FILENAME_TIME_SCALE = 10000.0


# ============================================================
# Paths
# ============================================================
base_root = r"E:/Rensq/DC_explore_F/data_1Tflow/"

final_save_root = (
    r"E:/Rensq/DC_explore_F/flow/"
    r"selected_8_cases_vorticity_contours"
)

os.makedirs(
    final_save_root,
    exist_ok=True
)

boundary_path = (
    r"E:/Rensq/DC_explore_F/flow/boundary.dat"
)


# ============================================================
# Plot domain
# ============================================================
flow_center_x = 0.0
flow_center_y = 0.0

x_min = -1.0
x_max = 20.0


def get_y_range_by_st(st):
    """
    根据 St 自动设置显示范围。
    """

    if st <= 0.5 + 1e-12:
        return -5.0, 5.0

    return -10.0, 10.0


wing_shift_x = flow_center_x
wing_shift_y = flow_center_y

x_ticks = np.array([
    0, 5, 10, 15, 20
])

y_tick_step = 5


def get_y_ticks_by_st(st):

    y_min_row, y_max_row = (
        get_y_range_by_st(st)
    )

    return np.arange(
        int(np.ceil(y_min_row)),
        int(np.floor(y_max_row)) + 1,
        y_tick_step
    )


Y_LABELPAD = -4


# ============================================================
# Downsample
# ============================================================
# 1 = 原始网格
# 2 = 每隔一个网格点绘制
#
# contour 图一般使用 2 即可。
PLOT_STRIDE = 2


# ============================================================
# Vorticity contour parameters
# ============================================================

# ------------------------------------------------------------
# Gaussian smoothing
#
# 主要用于去除网格尺度的小锯齿，使轮廓线平滑。
#
# 推荐：
# 1.0 ~ 2.0
# ------------------------------------------------------------
VORT_SMOOTH_SIGMA = 1.5


# ------------------------------------------------------------
# 用尾迹中的高百分位绝对涡量作为 omega_ref
#
# 不采用 max，避免个别极端点影响结果。
# ------------------------------------------------------------
VORT_REF_PERCENTILE = 99.5


# ------------------------------------------------------------
# 用于计算 omega_ref 的区域
#
# 后翼大约延伸至 x/c = 2.5，
# 因此从 2.6 开始统计真正的尾迹。
# ------------------------------------------------------------
VORT_REF_XMIN = 2.6
VORT_REF_XMAX = 19.5


# ------------------------------------------------------------
# 实际显示尾迹等值线的起始位置
#
# 设为 2.5 可以去掉翼面附近非常强、
# 非常密集的边界层和剪切层等值线。
#
# 若希望看更多近尾迹，可改为 2.0。
# ------------------------------------------------------------
WAKE_DRAW_XMIN = 2.5
WAKE_DRAW_XMAX = 20.0


# ------------------------------------------------------------
# 归一化等涡量线
#
# omega_n = omega / omega_ref
#
# 0.12：主要涡结构外围
# 0.30：较强涡核区域
#
# 推荐首先只保留两级，
# 比三四级轮廓更加简洁。
# ------------------------------------------------------------
VORT_LEVELS = [
    0.12,
    0.30,
]


# ------------------------------------------------------------
# 两层对应线宽
# ------------------------------------------------------------
VORT_LINEWIDTHS = [
    0.50,
    0.80,
]


# ------------------------------------------------------------
# 正负涡量区分
#
# True:
#   正涡量 -> 实线
#   负涡量 -> 虚线
#
# False:
#   正负全部实线
#
# 从尾迹分类角度推荐 True。
# ------------------------------------------------------------
DISTINGUISH_VORTICITY_SIGN = True


# ------------------------------------------------------------
# 是否删除很小的孤立涡结构
#
# 可以有效减少远场大量小圆圈。
# ------------------------------------------------------------
REMOVE_SMALL_COMPONENTS = True


# ------------------------------------------------------------
# 最小结构面积，单位 c^2
#
# 结构面积小于该值就删除。
#
# 越大：
#   图越干净，但可能删除真实小涡。
#
# 推荐：
# 0.01 ~ 0.05
# ------------------------------------------------------------
MIN_COMPONENT_AREA = 0.025


# ------------------------------------------------------------
# 背景透明
# ------------------------------------------------------------
TRANSPARENT_BACKGROUND = False


# ============================================================
# Airfoil profile
# ============================================================
profile = np.loadtxt(
    boundary_path
)

x_prof = profile[:, 0] * c
y_prof = profile[:, 1] * c


# ============================================================
# Utilities
# ============================================================
def fmt_num(x):
    """
    兼容 k2 与 k2.0 两类文件夹名称。
    """
    return f"{x:g}"


def unique_keep_order(seq):

    out = []

    for item in seq:

        if item not in out:
            out.append(item)

    return out


# ============================================================
# Find case directory
# ============================================================
def get_case_dir_candidates(
        base_root,
        k,
        st,
        a,
        dc):

    k_names = unique_keep_order([
        str(k),
        fmt_num(k)
    ])

    st_names = unique_keep_order([
        str(st),
        fmt_num(st)
    ])

    candidates = []

    for ks in k_names:

        for ss in st_names:

            case_dir_1 = os.path.join(
                base_root,

                f"k{ks}",

                f"st{ss}k{ks}",

                f"st{ss}k{ks}a{a}",

                f"st{ss}k{ks}a{a}dc{int(dc)}",

                f"k{ks}st{ss}"
            )

            case_dir_2 = os.path.join(
                base_root,

                f"k{ks}",

                f"st{ss}k{ks}",

                f"st{ss}k{ks}a{a}",

                f"st{ss}k{ks}a{a}dc{int(dc)}",

                f"k{ks}st{ss}a{a}"
            )

            candidates.append(
                case_dir_1
            )

            candidates.append(
                case_dir_2
            )

    return unique_keep_order(
        candidates
    )


def find_existing_case_dir(
        base_root,
        k,
        st,
        a,
        dc):

    candidates = (
        get_case_dir_candidates(
            base_root,
            k,
            st,
            a,
            dc
        )
    )

    for d in candidates:

        if os.path.isdir(d):
            return d

    return None


# ============================================================
# Tecplot ordered dat reader
# ============================================================
def read_tecplot_ordered_dat(
        filename,
        target_vars=(
            "X",
            "Y",
            "Vorticity"
        )):

    var_map = {
        "X": 0,
        "Y": 1,
        "U": 2,
        "V": 3,

        "Presure": 4,
        "Pressure": 4,

        "Vorticity": 5,
        "body": 6,
        "dvordt": 7
    }

    cols = [
        var_map[v]
        for v in target_vars
    ]

    need_cols = (
        max(cols) + 1
    )

    I = None
    J = None

    sol_time = None

    data_start_idx = None

    zone_found = False


    # --------------------------------------------------------
    # Read header
    # --------------------------------------------------------
    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()


    for idx, line_text in enumerate(
            lines,
            start=1):

        stripped = line_text.strip()

        if not stripped:
            continue


        # ----------------------------------------------------
        # ZONE
        # ----------------------------------------------------
        if (
            not zone_found
            and stripped.upper().startswith(
                "ZONE"
            )
        ):

            zone_found = True

            i_m = re.search(
                r"I\s*=\s*(\d+)",
                stripped,
                re.I
            )

            j_m = re.search(
                r"J\s*=\s*(\d+)",
                stripped,
                re.I
            )

            if i_m:
                I = int(
                    i_m.group(1)
                )

            if j_m:
                J = int(
                    j_m.group(1)
                )

            continue


        # ----------------------------------------------------
        # After ZONE
        # ----------------------------------------------------
        if zone_found:

            # solution time
            if sol_time is None:

                t_m = re.match(
                    r"^\s*([\d.Ee+\-]+)\s*$",
                    stripped
                )

                if t_m:

                    try:

                        sol_time = float(
                            t_m.group(1)
                        )

                        continue

                    except ValueError:
                        pass


            # first numerical data row
            parts = stripped.split()

            if len(parts) >= need_cols:

                try:

                    [
                        float(parts[c])
                        for c in cols
                    ]

                    data_start_idx = idx

                    break

                except ValueError:
                    continue


    if (
        I is None
        or J is None
        or data_start_idx is None
    ):

        raise ValueError(
            f"无法解析 dat 文件头信息: "
            f"{filename}"
        )


    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------
    data = np.loadtxt(
        filename,

        skiprows=(
            data_start_idx - 1
        ),

        max_rows=I * J,

        usecols=cols,

        delimiter=None
    )


    if data.shape[0] != I * J:

        raise ValueError(
            f"{filename} 数据点数量不足，"
            f"期望 {I * J}，"
            f"实际 {data.shape[0]}"
        )


    result = {
        "solution_time": sol_time,
        "I": I,
        "J": J
    }


    for i, var in enumerate(
            target_vars):

        result[var] = (
            data[:, i]
            .reshape(J, I)
        )


    return result


# ============================================================
# Filename time parser
# ============================================================
def parse_time_from_filename(
        filename):

    base = os.path.basename(
        filename
    )

    m = re.match(
        r"^F\s*(\d+)\.dat$",
        base,
        re.IGNORECASE
    )

    if not m:
        return None

    return (
        int(m.group(1))
        / FILENAME_TIME_SCALE
    )


def is_valid_flow_filename(
        filename):

    base = os.path.basename(
        filename
    )

    return (
        re.match(
            r"^F\s*\d+\.dat$",
            base,
            re.IGNORECASE
        )
        is not None
    )


# ============================================================
# Cycle / phase
# ============================================================
def compute_cycle_phase_from_time(
        t,
        T):

    ratio = t / T

    cycle_no = (
        int(
            np.floor(
                ratio + 1e-10
            )
        )
        + 1
    )

    phase = (
        ratio
        - np.floor(
            ratio + 1e-10
        )
    )

    if abs(phase) < 1e-8:
        phase = 0.0

    return cycle_no, phase


def find_target_record_by_k_cycle(
        case_dir,
        k,
        selected_cycle,
        target_phase=0.25,
        phase_tol=0.04):

    if (
        case_dir is None
        or
        not os.path.isdir(
            case_dir
        )
    ):

        return None


    T = np.pi / k

    target_time = (
        selected_cycle
        - 1
        + target_phase
    ) * T

    target_step = int(
        round(
            target_time
            * FILENAME_TIME_SCALE
        )
    )


    records = []


    for fname in os.listdir(
            case_dir):

        if not is_valid_flow_filename(
                fname):
            continue


        t = parse_time_from_filename(
            fname
        )

        if t is None:
            continue


        cycle_no, phase = (
            compute_cycle_phase_from_time(
                t,
                T
            )
        )


        records.append({
            "filename": fname,

            "path": os.path.join(
                case_dir,
                fname
            ),

            "t": t,

            "cycle_no": cycle_no,

            "phase_frac": phase,

            "time_error":
                abs(
                    t - target_time
                ),

            "phase_error":
                abs(
                    (t - target_time)
                    / T
                )
        })


    if not records:
        return None


    best = min(
        records,
        key=lambda r:
            r["time_error"]
    )


    if (
        best["phase_error"]
        <= phase_tol
    ):

        print(
            f"  target cycle="
            f"{selected_cycle}, "
            f"target phase="
            f"{target_phase:.3f}T"
        )

        print(
            f"  T={T:.6f}, "
            f"target t="
            f"{target_time:.6f}, "
            f"ideal "
            f"F{target_step:d}.dat"
        )

        print(
            f"  selected "
            f'{best["filename"]}, '
            f't={best["t"]:.6f}, '
            f'cycle='
            f'{best["cycle_no"]}, '
            f'phase='
            f'{best["phase_frac"]:.4f}T, '
            f'phase error='
            f'{best["phase_error"]:.4f}T'
        )

        return best


    print(
        f"  nearest file is "
        f'{best["filename"]}, '
        f"but phase error "
        f'{best["phase_error"]:.4f}T '
        f"> "
        f"PHASE_TOL="
        f"{phase_tol:.4f}T"
    )

    return None


# ============================================================
# Crop and downsample
# ============================================================
def crop_and_downsample(
        X,
        Y,
        W,
        x_min,
        x_max,
        y_min,
        y_max,
        stride=1):

    x_line = X[0, :]
    y_line = Y[:, 0]


    ix = np.where(
        (x_line >= x_min)
        &
        (x_line <= x_max)
    )[0]


    iy = np.where(
        (y_line >= y_min)
        &
        (y_line <= y_max)
    )[0]


    if (
        len(ix) == 0
        or
        len(iy) == 0
    ):

        raise ValueError(
            "指定绘图区间内没有网格点。"
        )


    i0 = ix[0]
    i1 = ix[-1] + 1

    j0 = iy[0]
    j1 = iy[-1] + 1


    Xc = X[
        j0:j1:stride,
        i0:i1:stride
    ]

    Yc = Y[
        j0:j1:stride,
        i0:i1:stride
    ]

    Wc = W[
        j0:j1:stride,
        i0:i1:stride
    ]


    return Xc, Yc, Wc


# ============================================================
# Remove small isolated structures
# ============================================================
def remove_small_components(
        X,
        Y,
        Wn,
        threshold,
        min_area):

    """
    删除面积很小的孤立高涡量区域。

    正、负涡量分别处理，避免相邻正负涡
    被视为同一个连通区域。
    """

    W_filtered = Wn.copy()


    # --------------------------------------------------------
    # Estimate grid cell area
    # --------------------------------------------------------
    if X.shape[1] > 1:

        dx = np.nanmedian(
            np.abs(
                np.diff(
                    X[0, :]
                )
            )
        )

    else:
        dx = 1.0


    if Y.shape[0] > 1:

        dy = np.nanmedian(
            np.abs(
                np.diff(
                    Y[:, 0]
                )
            )
        )

    else:
        dy = 1.0


    cell_area = dx * dy


    # 8-connected structure
    connectivity = np.ones(
        (3, 3),
        dtype=int
    )


    # --------------------------------------------------------
    # Positive vortices
    # --------------------------------------------------------
    positive_mask = (
        np.isfinite(W_filtered)
        &
        (W_filtered >= threshold)
    )


    labels_pos, n_pos = label(
        positive_mask,
        structure=connectivity
    )


    for idx in range(
            1,
            n_pos + 1):

        component = (
            labels_pos == idx
        )

        area = (
            np.count_nonzero(
                component
            )
            * cell_area
        )

        if area < min_area:

            W_filtered[
                component
            ] = 0.0


    # --------------------------------------------------------
    # Negative vortices
    # --------------------------------------------------------
    negative_mask = (
        np.isfinite(W_filtered)
        &
        (W_filtered <= -threshold)
    )


    labels_neg, n_neg = label(
        negative_mask,
        structure=connectivity
    )


    for idx in range(
            1,
            n_neg + 1):

        component = (
            labels_neg == idx
        )

        area = (
            np.count_nonzero(
                component
            )
            * cell_area
        )

        if area < min_area:

            W_filtered[
                component
            ] = 0.0


    return W_filtered


# ============================================================
# Vorticity contour plotting
# ============================================================
def plot_vorticity_contours(
        ax,
        X,
        Y,
        W,
        y_min_field,
        y_max_field):

    """
    绘制用于尾迹分类的归一化涡量等值线。

    处理流程：

        原始涡量
            ↓
        Gaussian 平滑
            ↓
        尾迹高百分位归一化
            ↓
        去除很小的孤立结构
            ↓
        正/负多级等值线
    """


    # --------------------------------------------------------
    # 1. Copy
    # --------------------------------------------------------
    W_work = np.asarray(
        W,
        dtype=float
    ).copy()


    W_work[
        ~np.isfinite(W_work)
    ] = 0.0


    # --------------------------------------------------------
    # 2. Gaussian smoothing
    # --------------------------------------------------------
    W_smooth = gaussian_filter(
        W_work,
        sigma=VORT_SMOOTH_SIGMA,
        mode="nearest"
    )


    # --------------------------------------------------------
    # 3. Reference region
    # --------------------------------------------------------
    ref_mask = (
        (X >= VORT_REF_XMIN)
        &
        (X <= VORT_REF_XMAX)
        &
        (Y >= y_min_field)
        &
        (Y <= y_max_field)
    )


    ref_values = np.abs(
        W_smooth[
            ref_mask
        ]
    )


    ref_values = ref_values[
        np.isfinite(
            ref_values
        )
    ]


    if ref_values.size == 0:

        print(
            "  No valid vorticity "
            "reference data."
        )

        return


    # --------------------------------------------------------
    # 4. Robust reference vorticity
    # --------------------------------------------------------
    omega_ref = np.percentile(
        ref_values,
        VORT_REF_PERCENTILE
    )


    if (
        not np.isfinite(omega_ref)
        or
        omega_ref <= 1e-12
    ):

        print(
            "  Invalid omega_ref."
        )

        return


    # --------------------------------------------------------
    # 5. Normalize
    # --------------------------------------------------------
    Wn = (
        W_smooth
        / omega_ref
    )


    print(
        f"  omega_ref = "
        f"{omega_ref:.6f}"
    )


    # --------------------------------------------------------
    # 6. Hide wing / near-body region
    # --------------------------------------------------------
    Wn_plot = Wn.copy()

    draw_mask = (
        (X >= WAKE_DRAW_XMIN)
        &
        (X <= WAKE_DRAW_XMAX)
        &
        (Y >= y_min_field)
        &
        (Y <= y_max_field)
    )


    Wn_plot[
        ~draw_mask
    ] = np.nan


    # --------------------------------------------------------
    # 7. Remove very small isolated structures
    # --------------------------------------------------------
    if REMOVE_SMALL_COMPONENTS:

        Wn_plot = (
            remove_small_components(
                X,
                Y,
                Wn_plot,
                threshold=min(
                    VORT_LEVELS
                ),
                min_area=(
                    MIN_COMPONENT_AREA
                )
            )
        )


    # --------------------------------------------------------
    # 8. Positive contours
    # --------------------------------------------------------
    positive_levels = np.array(
        VORT_LEVELS,
        dtype=float
    )


    valid_values = Wn_plot[
        np.isfinite(
            Wn_plot
        )
    ]


    if valid_values.size == 0:
        return


    max_value = np.max(
        valid_values
    )

    min_value = np.min(
        valid_values
    )


    # only use levels that exist
    pos_levels = (
        positive_levels[
            positive_levels
            <= max_value
        ]
    )


    if pos_levels.size > 0:

        pos_lws = np.asarray(
            VORT_LINEWIDTHS
        )[
            :pos_levels.size
        ]


        ax.contour(
            X,
            Y,
            Wn_plot,

            levels=pos_levels,

            colors="black",

            linewidths=pos_lws,

            linestyles="solid",

            antialiased=True,

            zorder=2
        )


    # --------------------------------------------------------
    # 9. Negative contours
    # --------------------------------------------------------
    neg_abs_levels = (
        positive_levels[
            positive_levels
            <= abs(
                min_value
            )
        ]
    )


    if neg_abs_levels.size > 0:

        # contour levels must be increasing
        neg_levels = (
            -neg_abs_levels[::-1]
        )


        neg_lws = (
            np.asarray(
                VORT_LINEWIDTHS
            )[
                :neg_abs_levels.size
            ][::-1]
        )


        if DISTINGUISH_VORTICITY_SIGN:

            neg_styles = "dashed"

        else:

            neg_styles = "solid"


        ax.contour(
            X,
            Y,
            Wn_plot,

            levels=neg_levels,

            colors="black",

            linewidths=neg_lws,

            linestyles=neg_styles,

            antialiased=True,

            zorder=2
        )


# ============================================================
# Wing plotting
# ============================================================
def plot_wing(
        ax,
        t,
        theta1,
        theta2,
        omega1,
        omega2,
        h0,
        alpha,
        phi,
        wing="fore",
        x_shift=0.0,
        y_shift=0.0,
        zorder=5,
        facecolor="white",
        edgecolor="black",
        lw=0.4):


    if wing == "fore":

        theta = (
            theta1
            * np.sin(
                omega1 * t
            )
        )

        h = (
            h0
            * np.sin(
                omega1 * t
                + phi
            )
        )

        x_le = 0.0


    else:

        theta = (
            theta2
            * np.sin(
                omega2 * t
            )
            - alpha
        )

        h = (
            h0
            * np.sin(
                omega2 * t
                + phi
            )
        )

        x_le = 1.5


    cos_th = np.cos(
        theta
    )

    sin_th = np.sin(
        theta
    )


    x_g = (
        x_prof * cos_th
        -
        y_prof * sin_th
        +
        x_le
        +
        x_shift
    )


    y_g = (
        x_prof * sin_th
        +
        y_prof * cos_th
        +
        h
        +
        y_shift
    )


    ax.fill(
        x_g,
        y_g,

        facecolor=facecolor,
        edgecolor=edgecolor,

        linewidth=lw,

        zorder=zorder
    )


# ============================================================
# Axis setup
# ============================================================
def setup_axis(
        ax,
        row_idx,
        col_idx,
        n_rows,
        n_cols,
        st):


    y_min_row, y_max_row = (
        get_y_range_by_st(st)
    )


    y_ticks_row = (
        get_y_ticks_by_st(st)
    )


    ax.set_xlim(
        x_min,
        x_max
    )


    ax.set_ylim(
        y_min_row,
        y_max_row
    )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.set_xticks(
        x_ticks
    )


    ax.set_yticks(
        y_ticks_row
    )


    ax.tick_params(
        direction="in",

        top=True,
        right=True,

        length=2.5,
        width=0.6
    )


    # --------------------------------------------------------
    # x axis
    # --------------------------------------------------------
    if row_idx == n_rows - 1:

        ax.tick_params(
            labelbottom=True
        )

        ax.set_xticklabels([
            r"$0$",
            r"$5$",
            r"$10$",
            r"$15$",
            r"$20$"
        ])

        ax.set_xlabel(
            r"$x/c$",
            labelpad=1
        )

    else:

        ax.tick_params(
            labelbottom=False
        )


    # --------------------------------------------------------
    # y axis
    # --------------------------------------------------------
    if col_idx == 0:

        ax.tick_params(
            labelleft=True
        )

        ax.set_ylabel(
            r"$y/c$",
            labelpad=Y_LABELPAD
        )

    else:

        ax.tick_params(
            labelleft=False
        )


    for spine in (
        ax.spines.values()
    ):

        spine.set_linewidth(
            0.6
        )


# ============================================================
# Build figure
# ============================================================
n_rows = 2
n_cols = 4


fig, axes = plt.subplots(
    n_rows,
    n_cols,

    figsize=(6.5, 2.5),

    dpi=300,

    sharex=True,
    sharey=False
)


plt.subplots_adjust(
    left=0.065,
    right=0.985,

    bottom=0.135,
    top=0.885,

    wspace=0.045,
    hspace=0.170
)


panel_labels = [
    "a", "b", "c", "d",
    "e", "f", "g", "h"
]


# ============================================================
# Main loop
# ============================================================
for case_idx, (k, st) in enumerate(
        SELECTED_CASES):


    row_idx = (
        case_idx
        // n_cols
    )


    col_idx = (
        case_idx
        % n_cols
    )


    ax = axes[
        row_idx,
        col_idx
    ]


    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------
    setup_axis(
        ax,
        row_idx,
        col_idx,
        n_rows,
        n_cols,
        st
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------
    ax.set_title(
        rf"$k={k:g},\ St={st:g}$",
        pad=4
    )


    print(
        "\n"
        "======================================"
    )

    print(
        f"Processing selected case: "
        f"k={k:g}, St={st:g}"
    )


    # --------------------------------------------------------
    # Find case directory
    # --------------------------------------------------------
    case_dir = (
        find_existing_case_dir(
            base_root,
            k,
            st,
            a,
            dc
        )
    )


    if case_dir is None:

        print(
            "  Case directory not found."
        )

        ax.text(
            0.5,
            0.5,
            "No data",

            transform=ax.transAxes,

            ha="center",
            va="center",

            fontsize=FONT_SIZE
        )

        continue


    # ========================================================
    # Wing kinematics
    # ========================================================
    theta1 = np.radians(
        30
    )

    theta2 = np.radians(
        30
    )


    f1 = k / np.pi
    f2 = f1


    h0 = (
        st * u
        /
        (2.0 * f1)
    )


    omega1 = (
        2.0
        * np.pi
        * f1
    )


    omega2 = (
        2.0
        * np.pi
        * f2
    )


    alpha = np.radians(
        a
    )


    phi = np.radians(
        90
    )


    # ========================================================
    # Find target phase
    # ========================================================
    rec = (
        find_target_record_by_k_cycle(
            case_dir=case_dir,

            k=k,

            selected_cycle=(
                TARGET_CYCLE
            ),

            target_phase=(
                TARGET_PHASE
            ),

            phase_tol=(
                PHASE_TOL
            )
        )
    )


    if rec is None:

        print(
            "  No suitable flow file."
        )

        ax.text(
            0.5,
            0.5,
            "No matched file",

            transform=ax.transAxes,

            ha="center",
            va="center",

            fontsize=FONT_SIZE
        )

        continue


    file_path = rec["path"]


    if not os.path.exists(
            file_path):

        print(
            f"  Missing file: "
            f"{file_path}"
        )

        ax.text(
            0.5,
            0.5,
            "Missing file",

            transform=ax.transAxes,

            ha="center",
            va="center",

            fontsize=FONT_SIZE
        )

        continue


    # ========================================================
    # Read and plot
    # ========================================================
    try:

        # ----------------------------------------------------
        # Only X, Y and vorticity are needed
        # ----------------------------------------------------
        dat = (
            read_tecplot_ordered_dat(
                file_path,

                target_vars=(
                    "X",
                    "Y",
                    "Vorticity"
                )
            )
        )


        X = dat["X"]
        Y = dat["Y"]
        W = dat["Vorticity"]


        # ----------------------------------------------------
        # Current y range
        # ----------------------------------------------------
        y_min_row, y_max_row = (
            get_y_range_by_st(
                st
            )
        )


        # ----------------------------------------------------
        # Crop + downsample
        # ----------------------------------------------------
        Xp, Yp, Wp = (
            crop_and_downsample(
                X,
                Y,
                W,

                x_min,
                x_max,

                y_min_row,
                y_max_row,

                stride=PLOT_STRIDE
            )
        )


        # ----------------------------------------------------
        # Vorticity skeleton
        # ----------------------------------------------------
        plot_vorticity_contours(
            ax,

            Xp,
            Yp,
            Wp,

            y_min_row,
            y_max_row
        )


        # ----------------------------------------------------
        # Wings
        # ----------------------------------------------------
        t = rec["t"]


        plot_wing(
            ax,
            t,

            theta1,
            theta2,

            omega1,
            omega2,

            h0,
            alpha,
            phi,

            wing="fore",

            x_shift=(
                wing_shift_x
            ),

            y_shift=(
                wing_shift_y
            ),

            edgecolor="black",
            facecolor="white",

            lw=0.4,

            zorder=10
        )


        plot_wing(
            ax,
            t,

            theta1,
            theta2,

            omega1,
            omega2,

            h0,
            alpha,
            phi,

            wing="hind",

            x_shift=(
                wing_shift_x
            ),

            y_shift=(
                wing_shift_y
            ),

            edgecolor="black",
            facecolor="white",

            lw=0.4,

            zorder=10
        )


    except Exception as e:

        print(
            f"  Read or plot failed:"
        )

        print(
            f"  {file_path}"
        )

        print(
            f"  Error: {e}"
        )


        ax.text(
            0.5,
            0.5,
            "Read failed",

            transform=ax.transAxes,

            ha="center",
            va="center",

            fontsize=FONT_SIZE
        )


        continue


# ============================================================
# Panel labels outside upper-left corner
# ============================================================
fig.canvas.draw()


# 数值越大越往左
LABEL_DX = 0.010

# 数值越大越往上
LABEL_DY = 0.008


for case_idx in range(
        len(SELECTED_CASES)):


    row_idx = (
        case_idx
        // n_cols
    )


    col_idx = (
        case_idx
        % n_cols
    )


    ax = axes[
        row_idx,
        col_idx
    ]


    pos = ax.get_position()


    # 括号保持正体，字母为斜体
    fig.text(
        pos.x0 - LABEL_DX,
        pos.y1 + LABEL_DY,

        rf"($\mathit{{"
        rf"{panel_labels[case_idx]}"
        rf"}}$)",

        ha="left",
        va="bottom",

        fontsize=FONT_SIZE
    )


# ============================================================
# Force all text objects to 10 pt
# ============================================================
for text in fig.findobj(
        match=plt.Text):

    text.set_fontsize(
        FONT_SIZE
    )


# ============================================================
# Save PNG
# ============================================================
final_png_name = (
    "selected_8_cases_"
    "normalized_vorticity_contours.png"
)


final_png_path = os.path.join(
    final_save_root,
    final_png_name
)


plt.savefig(
    final_png_path,

    format="png",

    bbox_inches="tight",

    dpi=600,

    transparent=(
        TRANSPARENT_BACKGROUND
    )
)


# ============================================================
# Save SVG
# ============================================================
final_svg_name = (
    "selected_8_cases_"
    "normalized_vorticity_contours.svg"
)


final_svg_path = os.path.join(
    final_save_root,
    final_svg_name
)


# SVG 不使用 bbox_inches='tight'，
# 避免 Illustrator 中布局发生变化。
plt.savefig(
    final_svg_path,

    format="svg",

    transparent=(
        TRANSPARENT_BACKGROUND
    )
)


plt.close(
    fig
)


print(
    "\n======================================"
)

print(
    "Finished."
)

print(
    f"PNG: {final_png_path}"
)

print(
    f"SVG: {final_svg_path}"
)