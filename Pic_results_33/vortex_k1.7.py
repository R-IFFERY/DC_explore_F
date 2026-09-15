# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import colormaps
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

    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # SVG 中字体保持为文字
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
# 基本参数
# ============================================================

u = 1.0
rou = 1.0
c = 1.0

# ============================================================
# 绘制工况设置
# ============================================================

ST = 0.3

# 每一行对应一个独立工况：
#
# (k, alpha_0, dc_code, dc_value, panel_letter)
#
# 第 1 行：(a) k=1.7, DC=0.75
# 第 2 行：(b) k=1.7, DC=0.50
# 第 3 行：(c) k=1.7, DC=0.25
# 第 4 行：(d) k=1.4, DC=0.25
#
ROW_CASES = [
    (1.7, 11, 75, 0.75, "a"),
    (1.7, 11, 50, 0.50, "b"),
    (1.7, 11, 25, 0.25, "c"),
    (1.4, 17, 25, 0.25, "d"),
]


# ============================================================
# 周期设置
# ============================================================

# 每一列对应一个周期
CYCLES = [
    25,
    26,
    27,
    28,
]

# 每个周期绘制 0.25T 相位
TARGET_PHASE = 0.25

# 允许保存流场与目标相位存在少量误差
PHASE_TOL = 0.04

# 每一列对应一个周期
CYCLES = [
    25,
    26,
    27,
    28,
]

# 每个周期绘制 0.25T 相位
TARGET_PHASE = 0.25

# 允许保存流场与目标相位存在少量误差
PHASE_TOL = 0.04


# ============================================================
# 数据路径
# ============================================================

base_root = r"E:/Rensq/DC_explore_F/data_1Tflow/"

final_save_root = (
    r"E:/Rensq/DC_explore_F/results_intermittent/vortex_dc_map_k1.7/"
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

y_min = -5.0
y_max = 5.0

wing_shift_x = flow_center_x
wing_shift_y = flow_center_y

x_ticks = np.array([
    0,
    5,
    10,
    15,
    20
])

y_ticks = np.array([
    -5,
    0,
    5
])

Y_LABELPAD = -4

# 流场降采样
# =1：原始分辨率
# =2：每两个网格绘制一次
PLOT_STRIDE = 2

# F165000.dat -> t = 16.5000
FILENAME_TIME_SCALE = 10000.0


# ============================================================
# Colormap
# ============================================================

numb = 512

base_cmap = (
    colormaps["RdBu_r"]
    .resampled(numb)
)

new_colors = base_cmap(
    np.linspace(
        0,
        1,
        numb
    )
)

vmin = -10
vmax = 10

# 让涡量接近 0 的区域趋于白色
sigma = 1.0

vals = np.linspace(
    vmin,
    vmax,
    numb
)

weights = np.exp(
    -0.5 * (vals / sigma) ** 2
)

for i in range(numb):

    w = weights[i]

    new_colors[i] = (
        (1 - w) * new_colors[i]
        +
        w * np.array([
            1,
            1,
            1,
            1
        ])
    )

cmap_white = mcolors.ListedColormap(
    new_colors
)

levels = np.linspace(
    vmin,
    vmax,
    numb + 1
)

norm = mcolors.Normalize(
    vmin=vmin,
    vmax=vmax
)


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
    数字转为适合文件夹名称的形式。

    例如：
        0.8 -> '0.8'
        1.0 -> '1'
    """

    if float(x).is_integer():
        return str(int(x))

    return f"{x:g}"


def unique_keep_order(seq):

    out = []

    for item in seq:

        if item not in out:
            out.append(item)

    return out


# ============================================================
# 工况路径
# ============================================================

def get_case_dir_candidates(
    base_root,
    k,
    st,
    a,
    dc_code,
    dc_value
):
    """
    当前主要针对 DC != 1。

    例如：

    E:/Rensq/DC_explore_F/data_1Tflow/
        k1.7/
        st0.3k1.7/
        st0.3k1.7a11/
        st0.3k1.7a11dc75/
        k1.7st0.3a11dc0.75/
    """

    k_names = unique_keep_order([
        str(k),
        fmt_num(k)
    ])

    st_names = unique_keep_order([
        str(st),
        fmt_num(st)
    ])

    a_names = unique_keep_order([
        str(a),
        fmt_num(a)
    ])

    dc_value_str = fmt_num(
        dc_value
    )

    candidates = []

    for ks in k_names:

        for ss in st_names:

            for aa in a_names:

                parent_dir = os.path.join(
                    base_root,
                    f"k{ks}",
                    f"st{ss}k{ks}",
                    f"st{ss}k{ks}a{aa}",
                    f"st{ss}k{ks}a{aa}dc{dc_code}"
                )

                # 标准间歇工况目录
                case_dir_1 = os.path.join(
                    parent_dir,
                    (
                        f"k{ks}"
                        f"st{ss}"
                        f"a{aa}"
                        f"dc{dc_value_str}"
                    )
                )

                # 备用目录形式
                case_dir_2 = os.path.join(
                    parent_dir,
                    f"k{ks}st{ss}a{aa}"
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
    dc_code,
    dc_value
):

    candidates = get_case_dir_candidates(
        base_root=base_root,
        k=k,
        st=st,
        a=a,
        dc_code=dc_code,
        dc_value=dc_value
    )

    for d in candidates:

        if os.path.isdir(d):
            return d

    print(
        "\n没有找到以下工况："
    )

    for d in candidates:
        print(
            f"    {d}"
        )

    return None


# ============================================================
# Tecplot dat reader
# ============================================================

def read_tecplot_ordered_dat(
    filename,
    target_vars=(
        "X",
        "Y",
        "Vorticity"
    )
):

    var_map = {
        "X": 0,
        "Y": 1,
        "U": 2,
        "V": 3,
        "Presure": 4,
        "Vorticity": 5,
        "body": 6,
        "dvordt": 7
    }

    cols = [
        var_map[v]
        for v in target_vars
    ]

    need_cols = (
        max(cols)
        + 1
    )

    I = None
    J = None

    sol_time = None
    data_start_idx = None

    zone_found = False

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        lines = f.readlines()

    # --------------------------------------------------------
    # 解析头部
    # --------------------------------------------------------

    for idx, line in enumerate(
        lines,
        start=1
    ):

        stripped = line.strip()

        if not stripped:
            continue

        if (
            not zone_found
            and
            stripped.upper().startswith("ZONE")
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
        or
        J is None
        or
        data_start_idx is None
    ):

        raise ValueError(
            f"无法解析 dat 文件头信息: "
            f"{filename}"
        )

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    data = np.loadtxt(
        filename,
        skiprows=data_start_idx - 1,
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
        target_vars
    ):

        result[var] = (
            data[:, i]
            .reshape(
                J,
                I
            )
        )

    return result


# ============================================================
# Flow filename utilities
# ============================================================

def parse_time_from_filename(
    filename
):
    """
    支持：

        F165000.dat
        F 165000.dat
        F    165000.dat
    """

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
        int(
            m.group(1)
        )
        /
        FILENAME_TIME_SCALE
    )


def is_valid_flow_filename(
    filename
):

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
# 周期与相位
# ============================================================

def compute_cycle_phase_from_time(
    t,
    T
):
    """
    周期编号从 1 开始。
    """

    cycle_no = (
        int(
            np.floor(
                t / T
                + 1e-10
            )
        )
        + 1
    )

    phase = (
        t / T
        -
        np.floor(
            t / T
            + 1e-10
        )
    )

    if abs(phase) < 1e-8:
        phase = 0.0

    return (
        cycle_no,
        phase
    )


# ============================================================
# 获取某工况所有流场记录
# ============================================================

def get_flow_records(
    case_dir,
    k
):

    if (
        case_dir is None
        or
        not os.path.isdir(case_dir)
    ):
        return []

    T = np.pi / k

    records = []

    for fname in os.listdir(
        case_dir
    ):

        if not is_valid_flow_filename(
            fname
        ):
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
        })

    records.sort(
        key=lambda r: r["t"]
    )

    return records


# ============================================================
# 指定周期寻找 0.25T
# ============================================================

def find_target_record_by_cycle(
    records,
    selected_cycle,
    target_phase=0.25,
    phase_tol=0.04
):

    cycle_records = [
        r
        for r in records
        if r["cycle_no"] == selected_cycle
    ]

    if len(cycle_records) == 0:

        print(
            f"    周期 {selected_cycle} "
            f"没有保存流场文件"
        )

        return None

    best = min(
        cycle_records,
        key=lambda r:
        abs(
            r["phase_frac"]
            - target_phase
        )
    )

    phase_error = abs(
        best["phase_frac"]
        - target_phase
    )

    if phase_error > phase_tol:

        print(
            f"    周期 {selected_cycle}: "
            f"最接近文件为 "
            f"{best['filename']}, "
            f"phase={best['phase_frac']:.4f}T, "
            f"误差={phase_error:.4f}T "
            f"> {phase_tol:.4f}T"
        )

        return None

    print(
        f"    cycle={selected_cycle}, "
        f"target={target_phase:.2f}T"
    )

    print(
        f"    -> {best['filename']}, "
        f"t={best['t']:.6f}, "
        f"phase={best['phase_frac']:.4f}T"
    )

    return best


# ============================================================
# 裁剪与降采样
# ============================================================

def crop_and_downsample(
    X,
    Y,
    W,
    x_min,
    x_max,
    y_min,
    y_max,
    stride=1
):

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

        return (
            X[::stride, ::stride],
            Y[::stride, ::stride],
            W[::stride, ::stride]
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

    return (
        Xc,
        Yc,
        Wc
    )


# ============================================================
# 从 force.dat 获取真实翼运动
# ============================================================

def read_wing_state_from_force(
    force_path,
    target_time
):
    """
    force.dat:

        0   time

        13  y1
        16  r1

        22  y2
        25  r2

    直接使用模拟程序保存的真实运动状态。
    """

    if not os.path.exists(
        force_path
    ):
        return None

    df = pd.read_csv(
        force_path,
        header=None,
        sep=r"\s+",
        engine="python",
        on_bad_lines="skip"
    )

    if df.shape[1] < 28:
        return None

    force_time = (
        pd.to_numeric(
            df.iloc[:, 0],
            errors="coerce"
        )
        .to_numpy()
    )

    valid = np.isfinite(
        force_time
    )

    if not np.any(valid):
        return None

    valid_indices = np.where(
        valid
    )[0]

    nearest_local = np.argmin(
        np.abs(
            force_time[valid]
            - target_time
        )
    )

    idx = valid_indices[
        nearest_local
    ]

    # 前翼
    h1 = float(
        df.iloc[
            idx,
            13
        ]
    )

    theta1 = float(
        df.iloc[
            idx,
            16
        ]
    )

    # 后翼
    h2 = float(
        df.iloc[
            idx,
            22
        ]
    )

    theta2 = float(
        df.iloc[
            idx,
            25
        ]
    )

    return {
        "time":
            force_time[idx],

        "h1":
            h1,

        "theta1":
            -theta1,

        "h2":
            h2,

        "theta2":
            -theta2,
    }


# ============================================================
# 绘制翼型
# ============================================================

def plot_wing_state(
    ax,
    theta,
    h,
    wing="fore",
    x_shift=0.0,
    y_shift=0.0,
    zorder=5,
    facecolor="white",
    edgecolor="black",
    lw=0.4
):

    if wing == "fore":

        x_le = 0.0

    else:

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

        facecolor=
            facecolor,

        edgecolor=
            edgecolor,

        linewidth=
            lw,

        zorder=
            zorder
    )


# ============================================================
# Axis
# ============================================================

def setup_axis(
    ax,
    row_idx,
    col_idx,
    n_rows,
    n_cols
):

    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.set_xticks(
        x_ticks
    )

    ax.set_yticks(
        y_ticks
    )

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=0.6
    )

    # --------------------------------------------------------
    # 横坐标
    # 仅最下面一行显示
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
    # 纵坐标
    # 仅第一列显示
    # --------------------------------------------------------

    if col_idx == 0:

        ax.tick_params(
            labelleft=True
        )

        ax.set_yticklabels([
            r"$-5$",
            r"$0$",
            r"$5$"
        ])

        ax.set_ylabel(
            r"$y/c$",
            labelpad=Y_LABELPAD
        )

    else:

        ax.tick_params(
            labelleft=False
        )

    for spine in ax.spines.values():

        spine.set_linewidth(
            0.6
        )


# ============================================================
# 绘制单个流场
# ============================================================

def plot_flow_snapshot(
    ax,
    rec,
    case_dir
):

    file_path = rec[
        "path"
    ]

    # --------------------------------------------------------
    # 读取流场
    # --------------------------------------------------------

    dat = read_tecplot_ordered_dat(
        file_path,
        target_vars=(
            "X",
            "Y",
            "Vorticity"
        )
    )

    X = dat[
        "X"
    ]

    Y = dat[
        "Y"
    ]

    W = dat[
        "Vorticity"
    ]

    # --------------------------------------------------------
    # 裁剪
    # --------------------------------------------------------

    Xp, Yp, Wp = crop_and_downsample(
        X,
        Y,
        W,
        x_min,
        x_max,
        y_min,
        y_max,
        stride=PLOT_STRIDE
    )

    Wp = np.ma.masked_where(
        (
            (Xp < x_min)
            |
            (Xp > x_max)
            |
            (Yp < y_min)
            |
            (Yp > y_max)
        ),
        Wp
    )

    # --------------------------------------------------------
    # 涡量
    # --------------------------------------------------------

    ax.contourf(
        Xp,
        Yp,
        Wp,

        levels=
            levels,

        cmap=
            cmap_white,

        extend=
            "both",

        antialiased=False
    )

    # --------------------------------------------------------
    # 根据 force.dat 绘制实际翼位置
    # --------------------------------------------------------

    force_path = os.path.join(
        case_dir,
        "force.dat"
    )

    wing_state = (
        read_wing_state_from_force(
            force_path,
            rec["t"]
        )
    )

    if wing_state is None:

        print(
            f"    [警告] 无法读取翼运动: "
            f"{force_path}"
        )

        return

    # 前翼
    plot_wing_state(
        ax=ax,

        theta=
            wing_state[
                "theta1"
            ],

        h=
            wing_state[
                "h1"
            ],

        wing="fore",

        x_shift=
            wing_shift_x,

        y_shift=
            wing_shift_y,

        edgecolor="black",

        lw=0.4,

        zorder=5
    )

    # 后翼
    plot_wing_state(
        ax=ax,

        theta=
            wing_state[
                "theta2"
            ],

        h=
            wing_state[
                "h2"
            ],

        wing="hind",

        x_shift=
            wing_shift_x,

        y_shift=
            wing_shift_y,

        edgecolor="black",

        lw=0.4,

        zorder=5
    )


# ============================================================
# Colorbar
# ============================================================

def add_colorbar(
    fig,
    host_ax
):

    cax = inset_axes(
        host_ax,

        width="55%",
        height="7%",

        loc="upper right",

        borderpad=0.45
    )

    sm = plt.cm.ScalarMappable(
        norm=norm,
        cmap=cmap_white
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        extend="both"
    )

    cbar.set_ticks([
        vmin,
        0,
        vmax
    ])

    cbar.set_ticklabels([
        rf"${vmin:g}$",
        r"$0$",
        rf"${vmax:g}$"
    ])

    cbar.ax.tick_params(
        labelsize=FONT_SIZE,
        length=0,
        width=0,
        pad=1
    )

    cbar.outline.set_linewidth(
        0.5
    )


# ============================================================
# 主绘图
# ============================================================

def plot_selected_dc_cycles():

    print(
        "\n"
        "============================================================"
    )

    print(
        "绘制指定间歇工况流场"
    )

    print(
        "Columns: cycles 25, 26, 27, 28"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # 图尺寸
    # ========================================================

    n_rows = len(
        ROW_CASES
    )

    n_cols = len(
        CYCLES
    )

    # --------------------------------------------------------
    # 创建 4 × 4 图
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        n_rows,
        n_cols,

        figsize=(
            6.5,
            3.8
        ),

        dpi=300,

        squeeze=False
    )

    plt.subplots_adjust(
        left=0.073,
        right=0.985,
        bottom=0.105,
        top=0.975,
        wspace=0.04,
        hspace=0.08
    )

    # ========================================================
    # 每一行对应一个独立工况
    # ========================================================

    for row_idx, (
        k,
        alpha_0,
        dc_code,
        dc_value,
        panel_letter
    ) in enumerate(
        ROW_CASES
    ):

        print(
            "\n"
            "------------------------------------------------------------"
        )

        print(
            f"Row {row_idx + 1}: "
            f"k={k:g}, "
            f"St={ST:g}, "
            f"alpha_0={alpha_0:g}, "
            f"DC={dc_value:g}"
        )

        # ----------------------------------------------------
        # 找工况目录
        # ----------------------------------------------------

        case_dir = (
            find_existing_case_dir(
                base_root=base_root,
                k=k,
                st=ST,
                a=alpha_0,
                dc_code=dc_code,
                dc_value=dc_value
            )
        )

        if case_dir is None:

            print(
                f"k={k:g}, "
                f"DC={dc_value:g} "
                f"数据目录不存在。"
            )

            # 即使数据不存在，也要设置坐标轴
            for col_idx in range(
                n_cols
            ):

                ax = axes[
                    row_idx,
                    col_idx
                ]

                setup_axis(
                    ax=ax,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    n_rows=n_rows,
                    n_cols=n_cols
                )

            # 添加子图编号
            first_ax = axes[
                row_idx,
                0
            ]

            first_ax.text(
                -0.26,
                0.94,

                rf"(\textit{{{panel_letter}}})",

                transform=first_ax.transAxes,

                ha="left",
                va="bottom",

                fontsize=FONT_SIZE,

                clip_on=False
            )

            continue

        print(
            f"directory: {case_dir}"
        )

        # ----------------------------------------------------
        # 一次性读取该工况下所有流场记录
        #
        # 注意：
        # 这里必须使用当前行自己的 k，
        # 因为周期 T = pi / k
        # ----------------------------------------------------

        records = get_flow_records(
            case_dir=case_dir,
            k=k
        )

        # ----------------------------------------------------
        # 每一列：
        #
        # cycle 25
        # cycle 26
        # cycle 27
        # cycle 28
        # ----------------------------------------------------

        for col_idx, cycle_no in enumerate(
            CYCLES
        ):

            ax = axes[
                row_idx,
                col_idx
            ]

            # ------------------------------------------------
            # 坐标轴
            # ------------------------------------------------

            setup_axis(
                ax=ax,
                row_idx=row_idx,
                col_idx=col_idx,
                n_rows=n_rows,
                n_cols=n_cols
            )

            print(
                f"\n"
                f"k={k:g}, "
                f"DC={dc_value:g}, "
                f"cycle={cycle_no}"
            )

            # ------------------------------------------------
            # 找到当前周期最接近 0.25T 的流场
            # ------------------------------------------------

            rec = find_target_record_by_cycle(
                records=records,
                selected_cycle=cycle_no,
                target_phase=TARGET_PHASE,
                phase_tol=PHASE_TOL
            )

            if rec is None:

                print(
                    f"    没有找到 "
                    f"cycle={cycle_no} "
                    f"的目标流场。"
                )

                continue

            # ------------------------------------------------
            # 绘制流场
            # ------------------------------------------------

            try:

                plot_flow_snapshot(
                    ax=ax,
                    rec=rec,
                    case_dir=case_dir
                )

            except Exception as e:

                print(
                    f"    Read/plot failed: "
                    f"{rec['path']}"
                )

                print(
                    f"    Error: {e}"
                )

        # ====================================================
        # 子图编号
        #
        # 每一整行只标一个：
        #
        # (a)
        # (b)
        # (c)
        # (d)
        #
        # 仅字母斜体
        # ====================================================

        first_ax = axes[
            row_idx,
            0
        ]

        first_ax.text(
            -0.26,
            0.94,

            rf"(\textit{{{panel_letter}}})",

            transform=first_ax.transAxes,

            ha="left",
            va="bottom",

            fontsize=FONT_SIZE,

            clip_on=False
        )

    # ========================================================
    # 色条
    #
    # 放在第一行最右侧图内右上角
    # ========================================================

    add_colorbar(
        fig,
        axes[0, -1]
    )

    # ========================================================
    # 强制所有字体为 10 pt
    # ========================================================

    for text in fig.findobj(
        match=plt.Text
    ):

        text.set_fontsize(
            FONT_SIZE
        )

    # ========================================================
    # 保存
    # ========================================================

    png_name = (
        "vortex_k17_k14_selected_DC_"
        "cycles25_28.png"
    )

    svg_name = (
        "vortex_k17_k14_selected_DC_"
        "cycles25_28.svg"
    )

    png_path = os.path.join(
        final_save_root,
        png_name
    )

    svg_path = os.path.join(
        final_save_root,
        svg_name
    )

    plt.savefig(
        png_path,
        format="png",
        bbox_inches="tight",
        dpi=600
    )

    plt.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        dpi=600
    )

    plt.close(
        fig
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {svg_path}"
    )

    print(
        "============================================================"
    )

if __name__ == "__main__":

    plot_selected_dc_cycles()