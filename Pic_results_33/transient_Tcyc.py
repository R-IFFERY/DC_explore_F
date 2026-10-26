# -*- coding: utf-8 -*-
"""
推进平衡状态下不同 DC 的瞬时系统升力/推力/功率系数。

布局：
    4 rows x 3 columns

    第一列：C_LS
    第二列：C_TS
    第三列：C_PS

每一行：
    k = 0.8, alpha0 = 20 deg
    k = 1.1, alpha0 = 20 deg
    k = 1.4, alpha0 = 17 deg
    k = 1.7, alpha0 = 11 deg

每个子图：
    DC = 1.00
    DC = 0.75
    DC = 0.50
    DC = 0.25

周期处理
------------------------------------------------------------
DC < 1：
    4 个普通周期 = 1 个大周期 4T

    使用五个完整大周期：
        13-16
        17-20
        21-24
        25-28
        29-32

    对每个大周期：
        1. 插值到统一 0~4T 相位网格
        2. 周期滑动平均
        3. 五个大周期逐相位平均
        4. 计算逐相位样本标准差

DC = 1：
    自动寻找 force.dat 最后四个完整普通周期，
    拼成一个 4T 区间。

    只进行滑动平均，
    不进行跨周期平均，
    不绘制误差带。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 基本参数
# ============================================================
ST = 0.3

BASE_ROOT = r"E:/Rensq/DC_explore_F/data_1Tflow"

SAVE_ROOT = (
    r"E:/Rensq/DC_explore_F/results_intermittent/"
    r"instantaneous_force_big_cycle"
)

os.makedirs(
    SAVE_ROOT,
    exist_ok=True
)


# ============================================================
# 推进平衡工况
# ============================================================
CASE_DICT = {
    0.8: 20,
    1.1: 20,
    1.4: 17,
    1.7: 11,
}


# ============================================================
# DC
# ============================================================
DC_LIST = [
    1.00,
    0.75,
    0.50,
    0.25,
]


# ============================================================
# DC < 1：
# 五个完整的 4T 大周期
# ============================================================
BIG_CYCLES = [
    (13, 16),
    (17, 20),
    (21, 24),
    (25, 28),
    (29, 32),
]


# ============================================================
# 时间及相位设置
# ============================================================
TIME_ORIGIN = 0.0

# 一个大周期包含4个普通周期
N_SMALL_CYCLES_PER_BIG = 4

# 每个普通周期插值后的相位点数
N_POINTS_PER_SMALL_CYCLE = 1000

# 一个4T大周期总点数
N_BIG_PHASE_POINTS = (
    N_SMALL_CYCLES_PER_BIG
    * N_POINTS_PER_SMALL_CYCLE
    + 1
)

# 内部计算相位：
# t/T = 0 ~ 4；绘图时进一步归一化为 t/T_cyc = 0 ~ 1
BIG_PHASE_GRID = np.linspace(
    0.0,
    4.0,
    N_BIG_PHASE_POINTS,
    endpoint=True
)

# ============================================================
# 绘图横坐标
#
# 一个大周期 T_cyc = 4T，因此将 4T 大周期归一化为 0~1：
#
#   0       -> 大周期起点
#   0.25    -> 第1个小周期 T 结束
#   0.50    -> 第2个小周期 T 结束
#   0.75    -> 第3个小周期 T 结束
#   1.00    -> 第4个小周期 T 结束
# ============================================================
PLOT_PHASE_GRID = (
    BIG_PHASE_GRID
    / N_SMALL_CYCLES_PER_BIG
)

# 每个普通周期 T 在大周期中的分界位置
SMALL_CYCLE_BOUNDARIES = [
    0.25,
    0.50,
    0.75,
]

MIN_SAMPLES_PER_SMALL_CYCLE = 10

# 样本标准差
STD_DDOF = 1


# ============================================================
# 滑动平均
# ============================================================

# 是否平滑
APPLY_SMOOTHING = True

# ------------------------------------------------------------
# 必须为奇数
#
# 41 点对应约 0.04T 的滑动窗口，
# 比之前的21点更平滑。
#
# 如果仍觉得曲线有小幅高频抖动，可以改成：
#
#     SMOOTH_WINDOW = 61
#
# 或：
#
#     SMOOTH_WINDOW = 81
#
# 一般不建议过大，否则会削弱瞬时力峰值。
# ------------------------------------------------------------
SMOOTH_WINDOW = 41


# ============================================================
# 绘图设置
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "font.family": "serif",
    "font.serif": ["Times New Roman"],

    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "text.latex.preamble": (
        r"\usepackage{amsmath}"
        r"\usepackage{newtxtext,newtxmath}"
    ),

    "axes.linewidth": 0.8,

    "xtick.direction": "in",
    "ytick.direction": "in",

    # 透明背景
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
})


# ============================================================
# DC 曲线样式
# ============================================================
DC_STYLE = {

    1.00: {
        "color": "black",
        "linestyle": "-",
        "label": r"$DC=1$",
    },

    0.75: {
        "color": "tab:blue",
        "linestyle": "-",
        "label": r"$DC=0.75$",
    },

    0.50: {
        "color": "tab:green",
        "linestyle": "-",
        "label": r"$DC=0.5$",
    },

    0.25: {
        "color": "tab:red",
        "linestyle": "-",
        "label": r"$DC=0.25$",
    },
}


# 平均曲线宽度
MEAN_LINEWIDTH = 0.7

# 误差带透明度
STD_ALPHA = 0.20

# y=0参考线
ZERO_LINEWIDTH = 0.60


# ============================================================
# 子图编号
# ============================================================
PANEL_LABELS = [
    r"($a$)",
    r"($b$)",
    r"($c$)",
    r"($d$)",
]


# ============================================================
# 坐标范围
# ============================================================

# 升力
CL_YLIM = (
    -5.0,
    12.0
)

CL_YTICKS = [
    -5.0,
    0.0,
    5.0,
    10.0,
]


# 推力
CT_YLIM = (
    -3.0,
    3.0
)

CT_YTICKS = [
    -2.0,
    0.0,
    2.0,
]

# 功率
# 默认根据全部工况的 C_PS 及误差带自动确定第三列统一纵坐标范围。
# 如需手动固定，可改为例如：
# CP_YLIM = (-5.0, 15.0)
CP_YLIM = (
    -2.0,
    8.0
)

CP_YTICKS = [
    -2.0,
    0.0,
    2.0,
    4.0,
    6.0,
    8.0
]



# ============================================================
# 数字格式
# ============================================================
def fmt_num(value):

    return f"{float(value):g}"


# ============================================================
# 构造 force.dat 路径
# ============================================================
def build_force_path(
    k,
    st,
    a,
    dc
):

    k_str = fmt_num(k)
    st_str = fmt_num(st)
    a_str = fmt_num(a)

    # ========================================================
    # DC = 1
    # ========================================================
    if np.isclose(
        dc,
        1.0
    ):

        dc_folder = "1"

        final_folder = (
            f"k{k_str}"
            f"st{st_str}"
            f"a{a_str}"
        )

    # ========================================================
    # DC < 1
    # ========================================================
    else:

        dc_code = int(
            round(
                dc * 100
            )
        )

        dc_folder = str(
            dc_code
        )

        dc_value_str = fmt_num(
            dc
        )

        final_folder = (
            f"k{k_str}"
            f"st{st_str}"
            f"a{a_str}"
            f"dc{dc_value_str}"
        )

    force_dir = os.path.join(

        BASE_ROOT,

        f"k{k_str}",

        f"st{st_str}k{k_str}",

        (
            f"st{st_str}"
            f"k{k_str}"
            f"a{a_str}"
        ),

        (
            f"st{st_str}"
            f"k{k_str}"
            f"a{a_str}"
            f"dc{dc_folder}"
        ),

        final_folder,
    )

    return os.path.join(
        force_dir,
        "force.dat"
    )


# ============================================================
# 读取 force.dat
# ============================================================
def read_force_data(
    force_path
):
    """
    读取 force.dat，并计算系统瞬时升力、推力和功率系数。

    关键列（Python 0-based）：
        0  : time
        2  : Fy1
        3  : M1
        5  : Fy2
        6  : M2
        7  : Fx_total
        8  : Fy_total
        14 : vy1
        17 : omega1
        23 : vy2
        26 : omega2

    系数定义：
        C_LS = Fy_total / 2
        C_TS = -Fx_total / 2

        C_P1 = -Fy1 * vy1 - M1 * omega1
        C_P2 = -Fy2 * vy2 - M2 * omega2
        C_PS = (C_P1 + C_P2) / 2
    """

    if not os.path.isfile(
        force_path
    ):
        raise FileNotFoundError(
            f"\n找不到 force.dat：\n"
            f"{force_path}"
        )

    df = pd.read_csv(
        force_path,
        header=None,
        sep=r"\s+",
        engine="python",
        on_bad_lines="skip"
    )

    if df.shape[1] < 27:
        raise ValueError(
            f"force.dat列数不足："
            f"{df.shape[1]}，计算功率至少需要27列。"
        )

    required_columns = [
        0,
        2, 3,
        5, 6,
        7, 8,
        14, 17,
        23, 26,
    ]

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    )

    time = df.iloc[:, 0].to_numpy(dtype=float)

    force_y1 = df.iloc[:, 2].to_numpy(dtype=float)
    moment1 = df.iloc[:, 3].to_numpy(dtype=float)

    force_y2 = df.iloc[:, 5].to_numpy(dtype=float)
    moment2 = df.iloc[:, 6].to_numpy(dtype=float)

    force_x_total = df.iloc[:, 7].to_numpy(dtype=float)
    force_y_total = df.iloc[:, 8].to_numpy(dtype=float)

    vy1 = df.iloc[:, 14].to_numpy(dtype=float)
    omega1 = df.iloc[:, 17].to_numpy(dtype=float)

    vy2 = df.iloc[:, 23].to_numpy(dtype=float)
    omega2 = df.iloc[:, 26].to_numpy(dtype=float)

    # ========================================================
    # 系统升力系数
    # ========================================================
    cl_total = (
        force_y_total
        / 2.0
    )

    # ========================================================
    # 系统推力系数
    # ========================================================
    ct_total = (
        -force_x_total
        / 2.0
    )

    # ========================================================
    # 系统功率系数
    # ========================================================
    cp1 = (
        -force_y1 * vy1
        - moment1 * omega1
    )

    cp2 = (
        -force_y2 * vy2
        - moment2 * omega2
    )

    cp_total = (
        cp1 + cp2
    ) / 2.0

    # --------------------------------------------------------
    # 按时间排序
    # --------------------------------------------------------
    order = np.argsort(
        time
    )

    time = time[order]
    cl_total = cl_total[order]
    ct_total = ct_total[order]
    cp_total = cp_total[order]

    # --------------------------------------------------------
    # 删除重复时间
    # --------------------------------------------------------
    (
        time,
        unique_indices
    ) = np.unique(
        time,
        return_index=True
    )

    cl_total = cl_total[unique_indices]
    ct_total = ct_total[unique_indices]
    cp_total = cp_total[unique_indices]

    return (
        time,
        cl_total,
        ct_total,
        cp_total
    )

# ============================================================
# 周期滑动平均
# ============================================================
def periodic_moving_average(
    signal,
    window_size
):
    """
    对4T大周期执行循环滑动平均。

    因为：
        t/T=0
    和
        t/T=4

    是相邻大周期的相同相位，因此采用循环 padding，
    避免普通滑动平均在左右端点处发生变形。
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    if window_size <= 1:

        return signal.copy()

    if window_size % 2 == 0:

        raise ValueError(
            "SMOOTH_WINDOW 必须为奇数。"
        )

    # --------------------------------------------------------
    # 去掉重复终点
    # --------------------------------------------------------
    core = signal[
        :-1
    ]

    if window_size >= len(
        core
    ):

        raise ValueError(
            "滑动窗口过大。"
        )

    half_window = (
        window_size
        // 2
    )

    # --------------------------------------------------------
    # 周期循环扩展
    # --------------------------------------------------------
    padded = np.pad(

        core,

        pad_width=(
            half_window,
            half_window
        ),

        mode="wrap"
    )

    kernel = (
        np.ones(
            window_size,
            dtype=float
        )
        / window_size
    )

    # --------------------------------------------------------
    # 滑动平均
    # --------------------------------------------------------
    smoothed_core = np.convolve(

        padded,

        kernel,

        mode="valid"
    )

    # --------------------------------------------------------
    # t/T=4 与 t/T=0 保持完全一致
    # --------------------------------------------------------
    smoothed = np.concatenate([
        smoothed_core,
        [
            smoothed_core[0]
        ]
    ])

    return smoothed


# ============================================================
# 判断一个普通周期是否完整
# ============================================================
def small_cycle_is_complete(
    time,
    cycle_number,
    period
):

    t_start = (
        TIME_ORIGIN
        + (
            cycle_number
            - 1
        )
        * period
    )

    t_end = (
        TIME_ORIGIN
        + cycle_number
        * period
    )

    tolerance = max(
        1.0e-10,
        period * 1.0e-8
    )

    mask = (
        (
            time
            >=
            t_start
            - tolerance
        )
        &
        (
            time
            <=
            t_end
            + tolerance
        )
    )

    temp_time = time[
        mask
    ]

    if len(
        temp_time
    ) < MIN_SAMPLES_PER_SMALL_CYCLE:

        return False

    phase = (
        temp_time
        - t_start
    ) / period

    phase_min = np.min(
        phase
    )

    phase_max = np.max(
        phase
    )

    return (
        phase_min
        <= 0.02
        and
        phase_max
        >= 0.98
    )


# ============================================================
# 找出全部完整普通周期
# ============================================================
def find_complete_small_cycles(
    time,
    period
):

    max_candidate = int(
        np.ceil(
            (
                np.max(time)
                - TIME_ORIGIN
            )
            / period
        )
    )

    complete_cycles = []

    for cycle_number in range(
        1,
        max_candidate + 1
    ):

        complete = (
            small_cycle_is_complete(
                time=time,
                cycle_number=cycle_number,
                period=period
            )
        )

        if complete:

            complete_cycles.append(
                cycle_number
            )

    return complete_cycles


# ============================================================
# 提取一个4T大周期
# ============================================================
def interpolate_big_cycle(
    time,
    signal,
    period,
    cycle_start,
    cycle_end
):

    n_small_cycles = (
        cycle_end
        - cycle_start
        + 1
    )

    if n_small_cycles != 4:

        raise ValueError(
            f"{cycle_start}-{cycle_end}"
            f"不包含4个普通周期。"
        )

    # ========================================================
    # 时间起点
    # ========================================================
    t_start = (
        TIME_ORIGIN
        + (
            cycle_start
            - 1
        )
        * period
    )

    # ========================================================
    # 时间终点
    # ========================================================
    t_end = (
        TIME_ORIGIN
        + cycle_end
        * period
    )

    tolerance = max(
        1.0e-10,
        period * 1.0e-8
    )

    mask = (
        (
            time
            >=
            t_start
            - tolerance
        )
        &
        (
            time
            <=
            t_end
            + tolerance
        )
    )

    block_time = time[
        mask
    ]

    block_signal = signal[
        mask
    ]

    # --------------------------------------------------------
    # 删除非有限值
    # --------------------------------------------------------
    valid = (
        np.isfinite(
            block_time
        )
        &
        np.isfinite(
            block_signal
        )
    )

    block_time = block_time[
        valid
    ]

    block_signal = block_signal[
        valid
    ]

    minimum_points = (
        MIN_SAMPLES_PER_SMALL_CYCLE
        * 4
    )

    if len(
        block_time
    ) < minimum_points:

        raise ValueError(
            f"大周期 "
            f"{cycle_start}-{cycle_end} "
            f"只有 {len(block_time)} 个数据点。"
        )

    # ========================================================
    # 以普通周期 T 归一化
    #
    # 横坐标 = 0 ~ 4
    # ========================================================
    block_phase = (
        block_time
        - t_start
    ) / period

    valid_phase = (
        (
            block_phase
            >= -1.0e-8
        )
        &
        (
            block_phase
            <= 4.0
            + 1.0e-8
        )
    )

    block_phase = block_phase[
        valid_phase
    ]

    block_signal = block_signal[
        valid_phase
    ]

    block_phase = np.clip(
        block_phase,
        0.0,
        4.0
    )

    # --------------------------------------------------------
    # 相位排序
    # --------------------------------------------------------
    order = np.argsort(
        block_phase
    )

    block_phase = block_phase[
        order
    ]

    block_signal = block_signal[
        order
    ]

    # --------------------------------------------------------
    # 删除重复相位
    # --------------------------------------------------------
    (
        block_phase,
        unique_indices
    ) = np.unique(
        block_phase,
        return_index=True
    )

    block_signal = block_signal[
        unique_indices
    ]

    # --------------------------------------------------------
    # 检查覆盖完整性
    # --------------------------------------------------------
    if (
        block_phase[0] > 0.02
        or
        block_phase[-1] < 3.98
    ):

        raise ValueError(
            f"大周期 "
            f"{cycle_start}-{cycle_end} "
            f"覆盖不完整："
            f"{block_phase[0]:.4f}"
            f" ~ "
            f"{block_phase[-1]:.4f}"
        )

    # ========================================================
    # 插值到统一大周期相位
    # ========================================================
    interpolated = np.interp(
        BIG_PHASE_GRID,
        block_phase,
        block_signal
    )

    # ========================================================
    # 滑动窗口平滑
    # ========================================================
    if APPLY_SMOOTHING:

        interpolated = (
            periodic_moving_average(
                interpolated,
                SMOOTH_WINDOW
            )
        )

    return interpolated


# ============================================================
# DC < 1
#
# 五个大周期：
# 均值 + 标准差
# ============================================================
def process_intermittent_signal(
    time,
    signal,
    k
):

    period = (
        np.pi
        / k
    )

    samples = []

    for (
        cycle_start,
        cycle_end
    ) in BIG_CYCLES:

        temp_signal = (
            interpolate_big_cycle(

                time=time,

                signal=signal,

                period=period,

                cycle_start=cycle_start,

                cycle_end=cycle_end
            )
        )

        samples.append(
            temp_signal
        )

    samples = np.asarray(
        samples,
        dtype=float
    )

    if samples.shape[0] != 5:

        raise ValueError(
            "没有获得完整的5个大周期。"
        )

    # ========================================================
    # 逐相位平均
    # ========================================================
    mean_signal = np.mean(
        samples,
        axis=0
    )

    # ========================================================
    # 逐相位样本标准差
    # ========================================================
    std_signal = np.std(
        samples,
        axis=0,
        ddof=STD_DDOF
    )

    return (
        mean_signal,
        std_signal,
        samples
    )


# ============================================================
# DC = 1：
# 找最后四个完整周期
# ============================================================
def find_last_four_cycles(
    time,
    k
):

    period = (
        np.pi
        / k
    )

    complete_cycles = (
        find_complete_small_cycles(
            time=time,
            period=period
        )
    )

    if len(
        complete_cycles
    ) < 4:

        raise ValueError(
            f"只有 "
            f"{len(complete_cycles)} "
            f"个完整周期。"
        )

    selected_cycles = (
        complete_cycles[
            -4:
        ]
    )

    if not np.all(
        np.diff(
            selected_cycles
        ) == 1
    ):

        raise ValueError(
            "最后四个完整周期不连续："
            f"{selected_cycles}"
        )

    return (
        selected_cycles,
        period
    )


# ============================================================
# 处理一个完整工况
# ============================================================
def process_one_case(
    k,
    a,
    dc
):

    force_path = build_force_path(
        k=k,
        st=ST,
        a=a,
        dc=dc
    )

    print(
        "\n"
        "------------------------------------------------------------"
    )

    print(
        f"k={k:g}, "
        f"St={ST:g}, "
        f"alpha0={a:g}, "
        f"DC={dc:g}"
    )

    print(
        f"读取：{force_path}"
    )

    (
        time,
        cl_total,
        ct_total,
        cp_total
    ) = read_force_data(
        force_path
    )

    result = {
        "k": k,
        "st": ST,
        "a": a,
        "dc": dc,
        "force_path": force_path,
    }

    # ========================================================
    # DC = 1
    # ========================================================
    if np.isclose(
        dc,
        1.0
    ):

        (
            selected_cycles,
            period
        ) = find_last_four_cycles(
            time=time,
            k=k
        )

        cycle_start = selected_cycles[0]
        cycle_end = selected_cycles[-1]

        print(
            "    DC=1使用最后四个完整周期："
            f"{cycle_start}-{cycle_end}"
        )

        cl_signal = interpolate_big_cycle(
            time=time,
            signal=cl_total,
            period=period,
            cycle_start=cycle_start,
            cycle_end=cycle_end
        )

        ct_signal = interpolate_big_cycle(
            time=time,
            signal=ct_total,
            period=period,
            cycle_start=cycle_start,
            cycle_end=cycle_end
        )

        cp_signal = interpolate_big_cycle(
            time=time,
            signal=cp_total,
            period=period,
            cycle_start=cycle_start,
            cycle_end=cycle_end
        )

        result.update({
            "cl_mean": cl_signal,
            "cl_std": None,

            "ct_mean": ct_signal,
            "ct_std": None,

            "cp_mean": cp_signal,
            "cp_std": None,

            "selected_cycles": selected_cycles,
        })

    # ========================================================
    # DC < 1
    # ========================================================
    else:

        (
            cl_mean,
            cl_std,
            cl_samples
        ) = process_intermittent_signal(
            time=time,
            signal=cl_total,
            k=k
        )

        (
            ct_mean,
            ct_std,
            ct_samples
        ) = process_intermittent_signal(
            time=time,
            signal=ct_total,
            k=k
        )

        (
            cp_mean,
            cp_std,
            cp_samples
        ) = process_intermittent_signal(
            time=time,
            signal=cp_total,
            k=k
        )

        result.update({
            "cl_mean": cl_mean,
            "cl_std": cl_std,
            "cl_samples": cl_samples,

            "ct_mean": ct_mean,
            "ct_std": ct_std,
            "ct_samples": ct_samples,

            "cp_mean": cp_mean,
            "cp_std": cp_std,
            "cp_samples": cp_samples,
        })

        print(
            "    采用的大周期："
            + ", ".join(
                [
                    f"{start}-{end}"
                    for (
                        start,
                        end
                    ) in BIG_CYCLES
                ]
            )
        )

        print(
            f"    CL std："
            f"mean={np.mean(cl_std):.4e}, "
            f"max={np.max(cl_std):.4e}"
        )

        print(
            f"    CT std："
            f"mean={np.mean(ct_std):.4e}, "
            f"max={np.max(ct_std):.4e}"
        )

        print(
            f"    CP std："
            f"mean={np.mean(cp_std):.4e}, "
            f"max={np.max(cp_std):.4e}"
        )

    return result

# ============================================================
# 绘制单个指标
# ============================================================
def draw_one_metric(
    ax,
    results_for_k,
    metric,
    ylabel,
    ylim=None,
    yticks=None
):

    # ========================================================
    # 先绘制误差带
    # ========================================================
    for dc in DC_LIST:

        result = results_for_k[
            dc
        ]

        mean_signal = result[
            f"{metric}_mean"
        ]

        std_signal = result[
            f"{metric}_std"
        ]

        style = DC_STYLE[
            dc
        ]

        # DC=1没有误差带
        if std_signal is not None:

            ax.fill_between(
                PLOT_PHASE_GRID,

                mean_signal
                - std_signal,

                mean_signal
                + std_signal,

                facecolor=style[
                    "color"
                ],

                edgecolor="none",

                linewidth=0.0,

                alpha=STD_ALPHA,

                interpolate=True,

                zorder=1
            )

    # ========================================================
    # 再绘制平均曲线
    # ========================================================
    for dc in DC_LIST:

        result = results_for_k[
            dc
        ]

        mean_signal = result[
            f"{metric}_mean"
        ]

        style = DC_STYLE[
            dc
        ]

        ax.plot(
            PLOT_PHASE_GRID,

            mean_signal,

            color=style[
                "color"
            ],

            linestyle=style[
                "linestyle"
            ],

            linewidth=MEAN_LINEWIDTH,

            label=style[
                "label"
            ],

            zorder=10
        )

    # ========================================================
    # y = 0
    # ========================================================
    ax.axhline(
        0.0,
        color="0.40",
        linewidth=ZERO_LINEWIDTH,
        linestyle=":",
        zorder=0
    )

    # ========================================================
    # 坐标范围
    # ========================================================
    ax.set_xlim(
        0.0,
        1.0
    )

    # ========================================================
    # 每个普通周期 T 的分界线
    #
    # 一个 T_cyc = 4T，因此分界位置为：
    # 0.25、0.50、0.75
    # ========================================================
    for x_boundary in SMALL_CYCLE_BOUNDARIES:

        ax.axvline(
            x=x_boundary,
            color="0.55",
            linewidth=0.55,
            linestyle="--",
            zorder=0
        )

    if ylim is not None:
        ax.set_ylim(
            ylim
        )

    if yticks is not None:
        ax.set_yticks(
            yticks
        )

    ax.set_ylabel(
        ylabel,
        rotation=0
    )

    ax.yaxis.set_label_coords(
        -0.12,
        0.5
    )

    ax.set_facecolor(
        "none"
    )

    ax.patch.set_alpha(
        0.0
    )

    ax.grid(
        False
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True
    )

# ============================================================
# 计算同一指标在全部工况中的统一纵坐标范围
# ============================================================
def get_global_metric_ylim(
    results,
    metric,
    pad_fraction=0.06
):

    value_min = np.inf
    value_max = -np.inf

    for k in CASE_DICT.keys():

        for dc in DC_LIST:

            result = results[
                k
            ][
                dc
            ]

            mean_signal = np.asarray(
                result[
                    f"{metric}_mean"
                ],
                dtype=float
            )

            std_signal = result[
                f"{metric}_std"
            ]

            if std_signal is None:

                lower = mean_signal
                upper = mean_signal

            else:

                std_signal = np.asarray(
                    std_signal,
                    dtype=float
                )

                lower = (
                    mean_signal
                    - std_signal
                )

                upper = (
                    mean_signal
                    + std_signal
                )

            value_min = min(
                value_min,
                np.nanmin(lower)
            )

            value_max = max(
                value_max,
                np.nanmax(upper)
            )

    # 保证 y=0 在图内
    value_min = min(
        value_min,
        0.0
    )

    value_max = max(
        value_max,
        0.0
    )

    data_range = (
        value_max
        - value_min
    )

    if not np.isfinite(
        data_range
    ):
        return None

    if data_range <= 0.0:
        data_range = max(
            abs(value_max),
            1.0
        )

    padding = (
        pad_fraction
        * data_range
    )

    return (
        value_min - padding,
        value_max + padding
    )


# ============================================================
# 绘制4 × 3总图
#
# 每一行作为一个逻辑子图：
#   (a) k=0.8
#   (b) k=1.1
#   (c) k=1.4
#   (d) k=1.7
#
# 三列依次：
#   C_LS, C_TS, C_PS
# ============================================================
def plot_all_results(
    results
):

    # 第三列 CP 使用全部工况统一纵坐标范围
    cp_ylim = (
        CP_YLIM
        if CP_YLIM is not None
        else get_global_metric_ylim(
            results=results,
            metric="cp"
        )
    )

    fig, axes = plt.subplots(
        4,
        3,

        figsize=(
            6.5,
            6.8
        ),

        sharex=True
    )

    fig.patch.set_facecolor(
        "none"
    )

    fig.patch.set_alpha(
        0.0
    )

    # ========================================================
    # 每一行一个 k
    # ========================================================
    for row_index, (
        k,
        a
    ) in enumerate(
        CASE_DICT.items()
    ):

        ax_cl = axes[
            row_index,
            0
        ]

        ax_ct = axes[
            row_index,
            1
        ]

        ax_cp = axes[
            row_index,
            2
        ]

        # ====================================================
        # 第一列：CL
        # ====================================================
        draw_one_metric(
            ax=ax_cl,
            results_for_k=results[k],
            metric="cl",
            ylabel=r"$C_{LS}$",
            ylim=CL_YLIM,
            yticks=CL_YTICKS
        )

        # ====================================================
        # 第二列：CT
        # ====================================================
        draw_one_metric(
            ax=ax_ct,
            results_for_k=results[k],
            metric="ct",
            ylabel=r"$C_{TS}$",
            ylim=CT_YLIM,
            yticks=CT_YTICKS
        )

        # ====================================================
        # 第三列：CP
        # ====================================================
        draw_one_metric(
            ax=ax_cp,
            results_for_k=results[k],
            metric="cp",
            ylabel=r"$C_{PS}$",
            ylim=cp_ylim,
            yticks=CP_YTICKS
        )

        # ====================================================
        # 每一行只标一个 (a)-(d)
        # 第二、第三列不再编号
        # ====================================================
        ax_cl.text(
            -0.23,
            1.02,

            PANEL_LABELS[
                row_index
            ],

            transform=ax_cl.transAxes,

            ha="left",
            va="bottom",

            fontsize=10,

            clip_on=False
        )

    # ========================================================
    # 横坐标
    # 最后一行三列均显示 t/T_cyc = 0~1
    # ========================================================
    for ax in axes[
        -1,
        :
    ]:

        ax.set_xlabel(
            r"$t/T_{\mathrm{cyc}}$"
        )

        ax.set_xticks([
            0.00,
            0.25,
            0.50,
            0.75,
            1.00
        ])

        ax.set_xticklabels([
            r"$0$",
            r"$0.25$",
            r"$0.5$",
            r"$0.75$",
            r"$1$"
        ])

    # ========================================================
    # 上三行不显示横坐标数字
    # ========================================================
    for row in range(
        3
    ):

        for col in range(
            3
        ):

            axes[
                row,
                col
            ].tick_params(
                labelbottom=False
            )

    # ========================================================
    # 图例
    # 只取曲线，不把误差带加入图例
    # ========================================================
    handles, labels = (
        axes[
            0,
            2
        ].get_legend_handles_labels()
    )

    axes[
        0,
        2
    ].legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.25,
        borderaxespad=0.4
    )

    # ========================================================
    # 布局
    # ========================================================
    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.075,
        top=0.985,
        wspace=0.20,
        hspace=0.12
    )

    # ========================================================
    # 保存
    # ========================================================
    output_base = os.path.join(
        SAVE_ROOT,
        "instantaneous_CL_CT_CP_big_cycle"
    )

    png_path = (
        output_base
        + ".png"
    )

    svg_path = (
        output_base
        + ".svg"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none"
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        transparent=True,
        facecolor="none",
        edgecolor="none"
    )

    plt.close(
        fig
    )

    print(
        f"\n图像已保存："
        f"{png_path}"
    )

    print(
        f"图像已保存："
        f"{svg_path}"
    )

# ============================================================
# 保存最终绘图数据
# ============================================================
def save_processed_data(
    results
):

    rows = []

    for (
        k,
        a
    ) in CASE_DICT.items():

        for dc in DC_LIST:

            result = results[
                k
            ][
                dc
            ]

            cl_std = result[
                "cl_std"
            ]

            ct_std = result[
                "ct_std"
            ]

            cp_std = result[
                "cp_std"
            ]

            for (
                index,
                phase
            ) in enumerate(
                BIG_PHASE_GRID
            ):

                rows.append({
                    "k": k,
                    "St": ST,
                    "alpha0": a,
                    "DC": dc,

                    "t_over_Tcyc":
                        (
                            phase
                            / N_SMALL_CYCLES_PER_BIG
                        ),

                    "CL":
                        result[
                            "cl_mean"
                        ][index],

                    "CL_std":
                        (
                            np.nan
                            if cl_std is None
                            else
                            cl_std[index]
                        ),

                    "CT":
                        result[
                            "ct_mean"
                        ][index],

                    "CT_std":
                        (
                            np.nan
                            if ct_std is None
                            else
                            ct_std[index]
                        ),

                    "CP":
                        result[
                            "cp_mean"
                        ][index],

                    "CP_std":
                        (
                            np.nan
                            if cp_std is None
                            else
                            cp_std[index]
                        ),
                })

    output_df = pd.DataFrame(
        rows
    )

    csv_path = os.path.join(
        SAVE_ROOT,
        "instantaneous_CL_CT_CP_big_cycle_data.csv"
    )

    output_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8e"
    )

    print(
        f"\n绘图数据已保存："
        f"{csv_path}"
    )

# ============================================================
# 主程序
# ============================================================
def main():

    results = {}

    # ========================================================
    # 数据处理
    # ========================================================
    for (
        k,
        a
    ) in CASE_DICT.items():

        results[
            k
        ] = {}

        for dc in DC_LIST:

            result = (
                process_one_case(
                    k=k,
                    a=a,
                    dc=dc
                )
            )

            results[
                k
            ][
                dc
            ] = result

    # ========================================================
    # 保存数据
    # ========================================================
    save_processed_data(
        results
    )

    # ========================================================
    # 绘制一张 4 × 3 图
    # ========================================================
    plot_all_results(
        results
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "全部处理完成"
    )

    print(
        f"保存目录："
        f"{SAVE_ROOT}"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
