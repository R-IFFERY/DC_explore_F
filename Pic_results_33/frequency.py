
# -*- coding: utf-8 -*-
"""
plot_POD_coefficients_spectrum_k1p4.py

固定工况：
    St = 0.3
    k = 1.4
    alpha0 = 17 deg

四行分别为：
    (a) DC = 1.00
    (b) DC = 0.75
    (c) DC = 0.50
    (d) DC = 0.25

每行：
    左：POD 时间系数 a1-a4
    右：a1-a4 的单边 FFT 幅值谱

时域横坐标：
    t/T_cyc = 0 ~ 1

频域横坐标：
    f_A/f

其中：
    f1 = 1/T

FFT处理：
    1. 使用完整4T大周期的128个snapshot
    2. 每个 ai 先去均值
    3. 不加窗
    4. 计算单边FFT幅值谱
    5. 同一DC的四个模态频谱统一除以四者在完整频段内的最大幅值
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径
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
# 2. 工况
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

N_MODES = 4


# ============================================================
# 3. 时域设置
# ============================================================

TIME_X_LIM = (
    0.0,
    1.0
)

TIME_X_TICKS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00
]

# 显式指定左图横坐标刻度文字，避免自动格式化为两位小数
TIME_X_TICKLABELS = [
    "0",
    "0.25",
    "0.5",
    "0.75",
    "1"
]

# 如果希望自动确定范围，改成 None
COEFF_Y_LIM = (
    -100.0,
    100.0
)

COEFF_Y_TICKS = [
    -100,
    0,
    100
]

COEFF_LINE_WIDTH = 0.7


# ============================================================
# 4. 频谱设置
# ============================================================

# f/f1显示范围
FREQ_X_LIM = (
    0.0,
    2.5
)

FREQ_X_TICKS = [
    0,
    0.25,
    0.5,
    0.75,
    1,
    2,
    # 3,
    # 4
]
FREQ_X_TICKLABELS = [
    "0",
    "0.25",
    "0.5",
    "0.75",
    "1",
    "2"
]

# 同一DC的四个模态共用完整频段内的最大幅值作为归一化分母
NORMALISE_SPECTRUM = True

SPECTRUM_Y_LIM = (
    0.0,
    1.05
)

SPECTRUM_Y_TICKS = [
    0,
    0.5,
    1
]

# 显式指定右图纵坐标刻度文字，避免显示成 0.0、1.0
SPECTRUM_Y_TICKLABELS = [
    "0",
    "0.5",
    "1"
]

SPECTRUM_LINE_WIDTH = 0.7


# ============================================================
# 关键频率
#
# 0.25 -> 4T
# 0.50 -> 2T
# 1.00 -> T
# ============================================================

SHOW_REFERENCE_FREQUENCIES = True

REFERENCE_FREQUENCIES = [
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# 5. 普通周期分界
# ============================================================

SHOW_CYCLE_DIVIDERS = True

CYCLE_DIVIDERS = [
    0.25,
    0.50,
    0.75,
]


# ============================================================
# 6. 图设置
# ============================================================

FONT_SIZE = 10

FIG_WIDTH = 6.5

# 每一行高度
ROW_HEIGHT = 1.15

# 上下边距
TOP_MARGIN = 0.40
BOTTOM_MARGIN = 0.45

# 行间距
ROW_GAP = 0.32

# 左右边距
LEFT_MARGIN = 0.72
RIGHT_MARGIN = 0.18

# 左右两图之间的距离
COLUMN_GAP = 0.55

# 左右图宽度比例
# 右侧频谱图略微加宽，左侧时域图相应略微缩窄
LEFT_WIDTH_RATIO = 1.15
RIGHT_WIDTH_RATIO = 1.05

DPI = 600

TRANSPARENT_BACKGROUND = True

OUT_BASENAME = (
    "Fig_POD_coeff_spectrum_k1p4_Tcyc"
)

# CSV输出文件名
SUMMARY_CSV_BASENAME = (
    "POD_coeff_spectrum_k1p4_key_parameters"
)

SPECTRUM_CSV_BASENAME = (
    "POD_coeff_spectrum_k1p4_full_spectrum"
)


# ============================================================
# 7. Matplotlib style
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
        10,

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
        0.9,

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
# 使用Matplotlib默认颜色循环
# 保证同一个ai在时域和频域中颜色一致
# ============================================================

DEFAULT_COLORS = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()[
        "color"
    ]
)

MODE_COLORS = [
    DEFAULT_COLORS[i]
    for i in range(N_MODES)
]

# 四个POD模态使用不同线型；
# 左侧时域图与右侧频谱图使用相同线型映射
MODE_LINESTYLES = [
    "-",    # a1: solid
    "--",   # a2: dashed
    "-",   # a3: dash-dot
    "--",    # a4: dotted
]


# ============================================================
# 8. Naming
# ============================================================

def fmt_num(
    x: float
) -> str:

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
    k,
    st,
    a,
    dc
):

    return (
        f"k{fmt_num_short(k)}"
        f"_St{fmt_num_short(st)}"
        f"_a{fmt_num_short(a)}"
        f"_dc{fmt_num_short(dc)}"
    )


def get_npz_path(
    dc
):

    cname = (
        case_name_from_values(
            K,
            ST,
            ALPHA0,
            dc
        )
    )

    return os.path.join(
        SAVE_ROOT,
        cname,
        f"{cname}_pod_results.npz"
    )


# ============================================================
# 9. 读取一个工况
# ============================================================

def load_case(
    dc
):

    npz_path = get_npz_path(
        dc
    )

    if not os.path.exists(
        npz_path
    ):

        raise FileNotFoundError(
            f"未找到：\n{npz_path}"
        )

    with np.load(
        npz_path
    ) as data:

        if "coeffs" not in data:

            raise KeyError(
                f"{npz_path}\n"
                "没有 coeffs"
            )

        coeffs = np.asarray(
            data[
                "coeffs"
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # 大周期相位
        # ----------------------------------------------------

        if "big_phase_frac" in data:

            big_phase = np.asarray(
                data[
                    "big_phase_frac"
                ],
                dtype=float
            )

        else:

            big_phase = None

    # ========================================================
    # 检查 coeffs
    # ========================================================

    if coeffs.ndim != 2:

        raise ValueError(
            f"coeffs shape = "
            f"{coeffs.shape}"
        )

    if coeffs.shape[0] < N_MODES:

        raise ValueError(
            f"coeffs只有"
            f"{coeffs.shape[0]}阶"
        )

    coeffs = coeffs[
        :N_MODES
    ].copy()

    n_snapshots = (
        coeffs.shape[1]
    )

    # ========================================================
    # 如果没有big_phase，自动生成
    # ========================================================

    if (
        big_phase is None
        or
        big_phase.size != n_snapshots
    ):

        print(
            f"警告：DC={dc:g} "
            "没有有效 big_phase_frac，"
            "使用等间距相位。"
        )

        big_phase = np.linspace(
            0.0,
            1.0,
            n_snapshots,
            endpoint=False
        )

    # ========================================================
    # 排序
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

    # ========================================================
    # big_phase:
    #
    #     0 ~ 1
    #
    # 转换为：
    #
    #     t/T = 0 ~ 4
    # ========================================================

    time_over_T = (
        4.0
        *
        big_phase
    )

    return {

        "dc":
            dc,

        "coeffs":
            coeffs,

        "big_phase":
            big_phase,

        "time_over_T":
            time_over_T,

        "npz_path":
            npz_path,
    }


# ============================================================
# 10. FFT前转为严格等间隔采样
# ============================================================

def make_uniform_time_series(
    time_over_T,
    coeffs
):

    time_over_T = np.asarray(
        time_over_T,
        dtype=float
    )

    coeffs = np.asarray(
        coeffs,
        dtype=float
    )

    n = time_over_T.size

    # ========================================================
    # 理论上一个4T大周期共有N个等间隔点：
    #
    # 0, 4/N, 2*4/N, ...
    # ========================================================

    uniform_time = np.linspace(
        0.0,
        4.0,
        n,
        endpoint=False
    )

    expected_dt = (
        4.0
        /
        n
    )

    if n > 1:

        actual_dt = np.diff(
            time_over_T
        )

        max_dt_error = np.nanmax(
            np.abs(
                actual_dt
                -
                expected_dt
            )
        )

    else:

        max_dt_error = 0.0

    # ========================================================
    # 如果已经基本等间距，直接使用
    # ========================================================

    if (
        np.all(
            np.isfinite(
                time_over_T
            )
        )
        and
        max_dt_error
        <
        1e-6
    ):

        return (
            uniform_time,
            coeffs.copy()
        )

    # ========================================================
    # 否则插值到严格等间距网格
    # ========================================================

    print(
        "警告：时间采样不是严格等间距，"
        "FFT前进行线性插值。"
    )

    coeffs_uniform = np.empty(
        (
            coeffs.shape[0],
            n
        ),
        dtype=float
    )

    for m in range(
        coeffs.shape[0]
    ):

        coeffs_uniform[m] = np.interp(
            uniform_time,
            time_over_T,
            coeffs[m]
        )

    return (
        uniform_time,
        coeffs_uniform
    )


# ============================================================
# 11. 单边FFT
# ============================================================

def calculate_fft(
    time_over_T,
    coeffs
):

    # ========================================================
    # 保证严格等间距
    # ========================================================

    (
        time_uniform,
        coeffs_uniform
    ) = make_uniform_time_series(
        time_over_T,
        coeffs
    )

    n = (
        coeffs_uniform.shape[1]
    )

    if n < 2:

        raise ValueError(
            "FFT数据点不足。"
        )

    # ========================================================
    # 一个snapshot的无量纲时间间隔
    #
    # 单位是 T
    # ========================================================

    dt_over_T = (
        4.0
        /
        n
    )

    # ========================================================
    # FFT frequency
    #
    # 因为时间单位是 T，
    # 所以这里的频率单位自动就是：
    #
    #     f / f1
    #
    # f1 = 1/T
    # ========================================================

    frequency = np.fft.rfftfreq(
        n,
        d=dt_over_T
    )

    # 始终保留未归一化的单边FFT幅值，用于CSV输出和定量比较
    raw_amplitudes = np.zeros(
        (
            N_MODES,
            frequency.size
        ),
        dtype=float
    )

    for m in range(
        N_MODES
    ):

        signal = (
            coeffs_uniform[m]
            .copy()
        )

        # ====================================================
        # 去均值
        #
        # 消除0频DC分量
        # ====================================================

        signal -= np.mean(
            signal
        )

        # ====================================================
        # FFT
        # ====================================================

        fft_values = np.fft.rfft(
            signal
        )

        # ====================================================
        # 单边幅值谱
        #
        # 首先除N
        # ====================================================

        amp = (
            np.abs(
                fft_values
            )
            /
            n
        )

        # ====================================================
        # 单边谱：
        #
        # 正频率需要乘2
        #
        # 0频不能乘2
        # Nyquist频率也不能乘2
        # ====================================================

        if n % 2 == 0:

            if amp.size > 2:

                amp[
                    1:-1
                ] *= 2.0

        else:

            if amp.size > 1:

                amp[
                    1:
                ] *= 2.0

        # ====================================================
        # 去均值后理论上f=0就是0
        # 为避免浮点残差，显式置0
        # ====================================================

        amp[0] = 0.0

        # ====================================================
        # 保存未归一化单边FFT幅值
        # ====================================================

        raw_amplitudes[m] = (
            amp.copy()
        )

    # ========================================================
    # 同一DC内的四个模态统一归一化。
    # 在四个模态的完整频段内取同一个最大幅值，保留模态间幅值比例。
    # 必须放在上述for循环结束后，待四个模态的原始频谱全部计算完成。
    # raw_amplitudes保持不变；amplitudes用于绘图及CSV的plotted字段。
    # ========================================================

    amplitudes = raw_amplitudes.copy()

    if NORMALISE_SPECTRUM:

        shared_max_amp = np.max(raw_amplitudes)

        if np.isfinite(shared_max_amp) and shared_max_amp > 0.0:

            amplitudes /= shared_max_amp

    return (
        time_uniform,
        coeffs_uniform,
        frequency,
        amplitudes,
        raw_amplitudes
    )


# ============================================================
# 12. Axis style
# ============================================================

def style_axis(
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
# 13. 绘制时域系数
# ============================================================

def plot_coefficients(
    ax,
    res,
    show_legend=False,
    show_xaxis=False
):

    time_over_Tcyc = (
        res[
            "time_uniform"
        ]
        / 4.0
    )

    coeffs = (
        res[
            "coeffs_uniform"
        ]
    )

    # ========================================================
    # 为绘图补上4T终点
    #
    # 第一个snapshot复制到4T处，
    # 使一个完整大周期闭合。
    # ========================================================

    time_plot = np.r_[
        time_over_Tcyc,
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

    for m in range(
        N_MODES
    ):

        ax.plot(
            time_plot,
            coeffs_plot[m],

            lw=
                COEFF_LINE_WIDTH,

            color=
                MODE_COLORS[m],

            ls=
                MODE_LINESTYLES[m],

            label=
                rf"$a_{m+1}$"
        )

    # ========================================================
    # y=0
    # ========================================================

    ax.axhline(
        0.0,
        color="0.68",
        lw=0.45,
        ls="--",
        zorder=0
    )

    # ========================================================
    # T, 2T, 3T
    # ========================================================

    if SHOW_CYCLE_DIVIDERS:

        for x in CYCLE_DIVIDERS:

            ax.axvline(
                x,
                color="0.76",
                lw=0.45,
                ls="--",
                zorder=0
            )

    # ========================================================
    # Axis
    # ========================================================

    ax.set_xlim(
        TIME_X_LIM
    )

    ax.set_xticks(
        TIME_X_TICKS
    )

    # 仅最后一行显示横坐标刻度值；并显式指定为
    # 0, 0.25, 0.5, 0.75, 1
    if show_xaxis:
        ax.set_xticklabels(
            TIME_X_TICKLABELS
        )
    else:
        ax.tick_params(
            axis="x",
            labelbottom=False
        )

    if COEFF_Y_LIM is not None:

        ax.set_ylim(
            COEFF_Y_LIM
        )

        ax.set_yticks(
            COEFF_Y_TICKS
        )

    if show_xaxis:
        ax.set_xlabel(
            r"$t/T_{\mathrm{cyc}}$",
            labelpad=-1.0
        )
    else:
        ax.set_xlabel("")

    ax.set_ylabel(
        r"$a_i$",
        rotation=0,
        ha="right",
        va="center",
        labelpad=-5
    )

    style_axis(
        ax
    )

    # ========================================================
    # Legend
    # ========================================================

    if show_legend:

        ax.legend(
            frameon=False,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.01
            ),
            handlelength=1.5,
            columnspacing=0.7,
            handletextpad=0.25,
            borderaxespad=0.1
        )


# ============================================================
# 14. 绘制频谱
# ============================================================

def plot_spectrum(
    ax,
    res,
    show_legend=False,
    show_xaxis=False
):

    frequency = (
        res[
            "frequency"
        ]
    )

    amplitudes = (
        res[
            "amplitudes"
        ]
    )

    # ========================================================
    # 仅绘制设定频率范围内的数据
    # ========================================================

    freq_mask = (
        (frequency >= FREQ_X_LIM[0])
        &
        (frequency <= FREQ_X_LIM[1])
    )

    for m in range(
        N_MODES
    ):

        ax.plot(
            frequency[
                freq_mask
            ],

            amplitudes[
                m,
                freq_mask
            ],

            lw=
                SPECTRUM_LINE_WIDTH,

            color=
                MODE_COLORS[m],

            ls=
                MODE_LINESTYLES[m],

            label=
                rf"$a_{m+1}$"
        )

    # ========================================================
    # 关键参考频率
    #
    # 0.25 -> 4T
    # 0.5  -> 2T
    # 1.0  -> T
    # ========================================================

    if SHOW_REFERENCE_FREQUENCIES:

        for f_ref in (
            REFERENCE_FREQUENCIES
        ):

            ax.axvline(
                f_ref,
                color="0.72",
                lw=0.5,
                ls="--",
                zorder=0
            )

    # ========================================================
    # Axis
    # ========================================================

    ax.set_xlim(
        FREQ_X_LIM
    )

    ax.set_xticks(
        FREQ_X_TICKS
    )

    # 仅最后一行显示频谱图横坐标刻度值
    # if not show_xaxis:
    #     ax.tick_params(
    #         axis="x",
    #         labelbottom=False
    #     )
    if show_xaxis:
        ax.set_xticklabels(
            FREQ_X_TICKLABELS
        )
    else:
        ax.tick_params(
            axis="x",
            labelbottom=False
        )

    if NORMALISE_SPECTRUM:

        ax.set_ylim(
            SPECTRUM_Y_LIM
        )

        ax.set_yticks(
            SPECTRUM_Y_TICKS
        )

        ax.set_yticklabels(
            SPECTRUM_Y_TICKLABELS
        )

        ax.set_ylabel(
            r"Amp*",
            labelpad=7.0,
            rotation=0,
        )

    else:

        ax.set_ylabel(
            r"Amp*",
            labelpad=8.0,
            rotation=0,
        )

    if show_xaxis:
        ax.set_xlabel(
            r"$f_a/f$",
            labelpad=-1.0
        )
    else:
        ax.set_xlabel("")

    style_axis(
        ax
    )

    if show_legend:

        ax.legend(
            frameon=False,
            ncol=1,
            loc="upper right",
            handlelength=1.5,
            handletextpad=0.35,
            labelspacing=0.35,
            borderaxespad=0.25
        )


# ============================================================
# 15. 手动布局
# ============================================================

def create_layout():

    nrows = len(
        DC_LIST
    )

    # ========================================================
    # 总高度
    # ========================================================

    fig_height = (
        TOP_MARGIN
        +
        BOTTOM_MARGIN
        +
        nrows
        *
        ROW_HEIGHT
        +
        (nrows - 1)
        *
        ROW_GAP
    )

    # ========================================================
    # 可用宽度
    # ========================================================

    usable_width = (
        FIG_WIDTH
        -
        LEFT_MARGIN
        -
        RIGHT_MARGIN
        -
        COLUMN_GAP
    )

    ratio_sum = (
        LEFT_WIDTH_RATIO
        +
        RIGHT_WIDTH_RATIO
    )

    left_width = (
        usable_width
        *
        LEFT_WIDTH_RATIO
        /
        ratio_sum
    )

    right_width = (
        usable_width
        *
        RIGHT_WIDTH_RATIO
        /
        ratio_sum
    )

    positions = []

    for i in range(
        nrows
    ):

        row_top = (
            fig_height
            -
            TOP_MARGIN
            -
            i
            *
            (
                ROW_HEIGHT
                +
                ROW_GAP
            )
        )

        row_bottom = (
            row_top
            -
            ROW_HEIGHT
        )

        left_pos = [

            LEFT_MARGIN
            /
            FIG_WIDTH,

            row_bottom
            /
            fig_height,

            left_width
            /
            FIG_WIDTH,

            ROW_HEIGHT
            /
            fig_height
        ]

        right_left = (
            LEFT_MARGIN
            +
            left_width
            +
            COLUMN_GAP
        )

        right_pos = [

            right_left
            /
            FIG_WIDTH,

            row_bottom
            /
            fig_height,

            right_width
            /
            FIG_WIDTH,

            ROW_HEIGHT
            /
            fig_height
        ]

        positions.append(
            (
                left_pos,
                right_pos
            )
        )

    return (
        fig_height,
        positions
    )


# ============================================================
# 16. Main
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "POD coefficient + FFT spectrum"
    )

    print(
        f"k={K:g}, "
        f"St={ST:g}, "
        f"alpha0={ALPHA0:g} deg"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # 读取全部DC
    # ========================================================

    results = []

    # 关键参数汇总：每个 DC × 每个 POD 模态一行
    summary_rows = []

    # 完整频谱：每个 DC × 每个 POD 模态 × 每个频率点一行
    spectrum_rows = []

    for dc in DC_LIST:

        print(
            f"\nLoading DC={dc:g}"
        )

        res = load_case(
            dc
        )

        (
            time_uniform,
            coeffs_uniform,
            frequency,
            amplitudes,
            raw_amplitudes
        ) = calculate_fft(

            res[
                "time_over_T"
            ],

            res[
                "coeffs"
            ]
        )

        res[
            "time_uniform"
        ] = time_uniform

        res[
            "coeffs_uniform"
        ] = coeffs_uniform

        res[
            "frequency"
        ] = frequency

        res[
            "amplitudes"
        ] = amplitudes

        res[
            "raw_amplitudes"
        ] = raw_amplitudes

        results.append(
            res
        )

        # ====================================================
        # 输出FFT基本信息
        # ====================================================

        n = (
            coeffs_uniform.shape[1]
        )

        delta_f = (
            frequency[1]
            -
            frequency[0]
        )

        print(
            f"  snapshots = {n}"
        )

        print(
            f"  frequency resolution "
            f"Delta(f/f1) = "
            f"{delta_f:.4f}"
        )

        print(
            f"  Nyquist = "
            f"{frequency[-1]:.4f}"
        )

        # ====================================================
        # 输出每个模态的关键参数，并写入CSV缓存
        # ====================================================

        for m in range(
            N_MODES
        ):

            mode_id = (
                m + 1
            )

            coeff = (
                coeffs_uniform[m]
                .copy()
            )

            amp_plot = (
                amplitudes[m]
                .copy()
            )

            amp_raw = (
                raw_amplitudes[m]
                .copy()
            )

            # 主频仅在实际绘图频率范围内搜索，并排除0频
            positive_mask = (
                (frequency > 0.0)
                &
                (frequency >= FREQ_X_LIM[0])
                &
                (frequency <= FREQ_X_LIM[1])
            )

            dominant_frequency = np.nan
            dominant_period_TA_over_T = np.nan
            dominant_amplitude_raw = np.nan
            dominant_amplitude_plot = np.nan

            if np.any(
                positive_mask
            ):

                freq_positive = (
                    frequency[
                        positive_mask
                    ]
                )

                amp_raw_positive = (
                    amp_raw[
                        positive_mask
                    ]
                )

                amp_plot_positive = (
                    amp_plot[
                        positive_mask
                    ]
                )

                max_index = np.argmax(
                    amp_raw_positive
                )

                dominant_frequency = (
                    freq_positive[
                        max_index
                    ]
                )

                dominant_amplitude_raw = (
                    amp_raw_positive[
                        max_index
                    ]
                )

                dominant_amplitude_plot = (
                    amp_plot_positive[
                        max_index
                    ]
                )

                if dominant_frequency > 0.0:

                    dominant_period_TA_over_T = (
                        1.0
                        /
                        dominant_frequency
                    )

                print(
                    f"    a{mode_id}: "
                    f"dominant f_A/f = "
                    f"{dominant_frequency:.3f}, "
                    f"A_raw = "
                    f"{dominant_amplitude_raw:.6g}"
                )

            # ------------------------------------------------
            # 时域系数统计量
            # ------------------------------------------------

            coeff_mean = float(
                np.mean(coeff)
            )

            coeff_min = float(
                np.min(coeff)
            )

            coeff_max = float(
                np.max(coeff)
            )

            coeff_peak_to_peak = float(
                np.ptp(coeff)
            )

            coeff_rms = float(
                np.sqrt(
                    np.mean(
                        coeff ** 2
                    )
                )
            )

            coeff_std = float(
                np.std(
                    coeff,
                    ddof=0
                )
            )

            # ------------------------------------------------
            # 关键参数汇总表
            # ------------------------------------------------

            summary_rows.append(
                {
                    "k": K,
                    "St": ST,
                    "alpha0_deg": ALPHA0,
                    "DC": dc,
                    "mode": mode_id,
                    "n_snapshots": n,
                    "frequency_resolution_fA_over_f": float(delta_f),
                    "nyquist_fA_over_f": float(frequency[-1]),
                    "coeff_mean": coeff_mean,
                    "coeff_min": coeff_min,
                    "coeff_max": coeff_max,
                    "coeff_peak_to_peak": coeff_peak_to_peak,
                    "coeff_rms": coeff_rms,
                    "coeff_std": coeff_std,
                    "dominant_frequency_fA_over_f": float(dominant_frequency),
                    "dominant_period_TA_over_T": float(dominant_period_TA_over_T),
                    "dominant_fft_amplitude_raw": float(dominant_amplitude_raw),
                    "dominant_fft_amplitude_plotted": float(dominant_amplitude_plot),
                    "spectrum_normalised_for_plot": NORMALISE_SPECTRUM,
                }
            )

            # ------------------------------------------------
            # 完整频谱表
            # ------------------------------------------------

            for j in range(
                frequency.size
            ):

                spectrum_rows.append(
                    {
                        "k": K,
                        "St": ST,
                        "alpha0_deg": ALPHA0,
                        "DC": dc,
                        "mode": mode_id,
                        "frequency_fA_over_f": float(frequency[j]),
                        "fft_amplitude_raw": float(amp_raw[j]),
                        "fft_amplitude_plotted": float(amp_plot[j]),
                    }
                )

    # ========================================================
    # Layout
    # ========================================================

    (
        fig_height,
        positions
    ) = create_layout()

    print(
        "\n"
        f"Figure size = "
        f"{FIG_WIDTH:.3f} × "
        f"{fig_height:.3f} in"
    )

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

    time_axes = []
    spectrum_axes = []

    # ========================================================
    # Draw
    # ========================================================

    for i, res in enumerate(
        results
    ):

        left_pos, right_pos = (
            positions[i]
        )

        ax_time = fig.add_axes(
            left_pos,
            facecolor="none"
        )

        ax_freq = fig.add_axes(
            right_pos,
            facecolor="none"
        )

        ax_time.patch.set_alpha(
            0.0
        )

        ax_freq.patch.set_alpha(
            0.0
        )

        time_axes.append(
            ax_time
        )

        spectrum_axes.append(
            ax_freq
        )

        # ====================================================
        # Time coefficients
        # ====================================================

        plot_coefficients(
            ax_time,
            res,
            show_legend=False,
            show_xaxis=(
                i == len(results) - 1
            )
        )

        # ====================================================
        # Frequency spectrum
        # ====================================================

        plot_spectrum(
            ax_freq,
            res,
            show_legend=(
                i == 0
            ),
            show_xaxis=(
                i == len(results) - 1
            )
        )

    # ========================================================
    # 主面板标签：
    #
    # (a) DC=1
    # (b) DC=0.75
    # (c) DC=0.5
    # (d) DC=0.25
    # ========================================================

    for i, res in enumerate(
        results
    ):

        panel_letter = chr(
            ord("a")
            +
            i
        )

        ax = (
            time_axes[i]
        )

        pos = (
            ax.get_position()
        )

        fig.text(
            pos.x0 - 0.065,
            pos.y1 + 0.004,

            (
                rf"$({panel_letter})$"
                # r"\quad"
                # rf"$DC={res['dc']:g}$"
            ),

            ha="left",
            va="bottom",

            fontsize=
                FONT_SIZE
        )

    # ========================================================
    # 列标题
    # ========================================================

    # first_time_pos = (
    #     time_axes[0]
    #     .get_position()
    # )
    #
    # first_freq_pos = (
    #     spectrum_axes[0]
    #     .get_position()
    # )
    #
    # fig.text(
    #     (
    #         first_time_pos.x0
    #         +
    #         first_time_pos.x1
    #     )
    #     /
    #     2.0,
    #
    #     first_time_pos.y1
    #     +
    #     0.045,
    #
    #     r"Temporal coefficients",
    #
    #     ha="center",
    #     va="bottom",
    #     fontsize=
    #         FONT_SIZE
    # )
    #
    # fig.text(
    #     (
    #         first_freq_pos.x0
    #         +
    #         first_freq_pos.x1
    #     )
    #     /
    #     2.0,
    #
    #     first_freq_pos.y1
    #     +
    #     0.045,
    #
    #     r"Frequency spectrum",
    #
    #     ha="center",
    #     va="bottom",
    #     fontsize=
    #         FONT_SIZE
    # )

    # ========================================================
    # Save
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

    summary_csv_path = os.path.join(
        FIG_SAVE_DIR,
        SUMMARY_CSV_BASENAME
        +
        ".csv"
    )

    spectrum_csv_path = os.path.join(
        FIG_SAVE_DIR,
        SPECTRUM_CSV_BASENAME
        +
        ".csv"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
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
        edgecolor="none"
    )

    fig.savefig(
        svg_path,
        dpi=DPI,
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
        edgecolor="none"
    )

    # ========================================================
    # 保存CSV
    # ========================================================

    if summary_rows:

        with open(
            summary_csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    summary_rows[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

    if spectrum_rows:

        with open(
            spectrum_csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    spectrum_rows[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(
                spectrum_rows
            )

    plt.close(
        fig
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "绘图完成"
    )

    print(
        png_path
    )

    print(
        svg_path
    )

    print(
        summary_csv_path
    )

    print(
        spectrum_csv_path
    )

    print(
        "=" * 72
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
