# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================================================
# 绘图设置
# ============================================================

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,

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

    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
})


# ============================================================
# 数据路径
# ============================================================

data_root = (
    r'E:/Rensq/DC_explore_F/results_intermittent/'
    r'averaged_results/'
)

save_root = os.path.join(
    data_root,
    'large_cycle_stacked_results'
)

os.makedirs(
    save_root,
    exist_ok=True
)


# ============================================================
# 工况设置
# ============================================================

ST = 0.3

# k : alpha_0
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
# DC 设置
# ============================================================

DC_CODES = [
    1,
    75,
    50,
    25,
]

DC_VALUE_DICT = {
    1: 1.00,
    75: 0.75,
    50: 0.50,
    25: 0.25,
}

DC_PLOT_ORDER = [
    1,
    75,
    50,
    25,
]


# ============================================================
# DC < 1 的大周期
# ============================================================

BIG_CYCLES = [
    (13, 16),
    (17, 20),
    (21, 24),
    (25, 28),
    (29, 32),
]

REQUIRE_COMPLETE_BIG_CYCLE = True


# ============================================================
# DC = 1
# 使用最后 5 个普通周期
# ============================================================

DC1_LAST_N_PERIODS = 5


# ============================================================
# 绘图参数
# ============================================================

BAR_WIDTH = 0.58

# 前翼浅灰，后翼深灰
FORE_COLOR = '0.82'
HIND_COLOR = '0.48'

BAR_EDGE_COLOR = 'black'
BAR_LINEWIDTH = 0.7

SYSTEM_LINE_COLOR = 'black'
SYSTEM_LINEWIDTH = 0.9
SYSTEM_MARKERSIZE = 4.2


# ============================================================
# 数字格式
# ============================================================

def format_num(x):

    if float(x).is_integer():
        return str(int(x))

    return f'{x:g}'


# ============================================================
# CSV 路径
# ============================================================

def get_csv_path(
    k,
    a,
    dc_code
):

    k_str = format_num(k)
    st_str = format_num(ST)
    a_str = format_num(a)

    filename = (
        f'k={k_str}'
        f'st={st_str}'
        f'a={a_str}'
        f'dc={dc_code}.csv'
    )

    return os.path.join(
        data_root,
        filename
    )


# ============================================================
# 计算一个统计单元
#
# 原始数据定义：
#
# C_LS = (C_LF + C_LH) / 2
# C_PS = (C_PF + C_PH) / 2
#
# 升力效率定义：
#
# eta_LF = C_LF / C_PF
# eta_LH = C_LH / C_PH
#
# eta_LS = (C_LF + C_LH) / (C_PF + C_PH)
#
# ============================================================

def calculate_values(temp):

    # ========================================================
    # 升力
    # ========================================================

    cl_fore = temp[
        'cl1_mean'
    ].mean()

    cl_hind = temp[
        'cl2_mean'
    ].mean()

    cl_system = temp[
        'cl_total_mean'
    ].mean()


    # ========================================================
    # 功率
    # ========================================================

    cp_fore = temp[
        'cp1_mean'
    ].mean()

    cp_hind = temp[
        'cp2_mean'
    ].mean()

    cp_system = temp[
        'cp_total_mean'
    ].mean()


    # ========================================================
    # 前翼升力效率
    #
    # eta_LF = C_LF / C_PF
    # ========================================================

    if np.isclose(
        cp_fore,
        0.0
    ):

        eta_l_fore = np.nan

    else:

        eta_l_fore = (
            cl_fore /
            cp_fore
        )


    # ========================================================
    # 后翼升力效率
    #
    # eta_LH = C_LH / C_PH
    # ========================================================

    if np.isclose(
        cp_hind,
        0.0
    ):

        eta_l_hind = np.nan

    else:

        eta_l_hind = (
            cl_hind /
            cp_hind
        )


    # ========================================================
    # 系统升力效率
    #
    # eta_LS
    # = (C_LF + C_LH)
    #   -----------------
    #   (C_PF + C_PH)
    #
    # 注意：
    # 虽然在当前归一化方式下也等价于 C_LS/C_PS，
    # 这里直接按定义使用前、后翼之和计算。
    # ========================================================

    cl_sum = (
        cl_fore +
        cl_hind
    )

    cp_sum = (
        cp_fore +
        cp_hind
    )

    if np.isclose(
        cp_sum,
        0.0
    ):

        eta_l_system = np.nan

    else:

        eta_l_system = (
            cl_sum /
            cp_sum
        )


    return {

        'cl_fore':
            cl_fore,

        'cl_hind':
            cl_hind,

        'cl_system':
            cl_system,

        'cp_fore':
            cp_fore,

        'cp_hind':
            cp_hind,

        'cp_system':
            cp_system,

        'eta_l_fore':
            eta_l_fore,

        'eta_l_hind':
            eta_l_hind,

        'eta_l_system':
            eta_l_system,
    }


# ============================================================
# DC = 1
#
# 最后 5 个普通周期分别作为统计样本
# ============================================================

def process_dc1(
    df,
    k,
    a,
    dc_code
):

    available_periods = sorted(
        df[
            'period'
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    if len(
        available_periods
    ) < DC1_LAST_N_PERIODS:

        print(
            f'[警告] '
            f'k={k}, '
            f'DC=1 '
            f'只有 {len(available_periods)} 个有效周期，'
            f'不足 {DC1_LAST_N_PERIODS} 个'
        )

        return None


    selected_periods = (
        available_periods[
            -DC1_LAST_N_PERIODS:
        ]
    )


    print(
        f'    DC=1 使用最后 '
        f'{DC1_LAST_N_PERIODS} 个周期: '
        f'{selected_periods}'
    )


    results = []


    for (
        sample_index,
        period_idx
    ) in enumerate(
        selected_periods,
        start=1
    ):

        temp = df[
            df[
                'period'
            ] == period_idx
        ].copy()


        if len(temp) == 0:
            continue


        values = calculate_values(
            temp
        )


        row = {

            'k':
                k,

            'St':
                ST,

            'a':
                a,

            'dc_code':
                dc_code,

            'dc':
                DC_VALUE_DICT[
                    dc_code
                ],

            'sample_index':
                sample_index,

            'sample_type':
                'single_cycle',

            'period_start':
                period_idx,

            'period_end':
                period_idx,

            'sample_label':
                f'{period_idx}',

            'n_base_cycles':
                1,
        }


        row.update(
            values
        )

        results.append(
            row
        )


    if len(results) == 0:
        return None


    return pd.DataFrame(
        results
    )


# ============================================================
# DC < 1
#
# 每一个 4T 大周期作为独立统计样本
# ============================================================

def process_intermittent_dc(
    df,
    k,
    a,
    dc_code
):

    results = []


    for (
        sample_index,
        (
            p_start,
            p_end
        )
    ) in enumerate(
        BIG_CYCLES,
        start=1
    ):

        expected_periods = list(
            range(
                p_start,
                p_end + 1
            )
        )


        temp = df[
            df[
                'period'
            ].isin(
                expected_periods
            )
        ].copy()


        actual_periods = sorted(
            temp[
                'period'
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )


        # ----------------------------------------------------
        # 要求完整 4T
        # ----------------------------------------------------

        if REQUIRE_COMPLETE_BIG_CYCLE:

            if actual_periods != expected_periods:

                print(
                    f'[警告] '
                    f'k={k}, '
                    f'DC={DC_VALUE_DICT[dc_code]}: '
                    f'{p_start}-{p_end} 周期不完整'
                )

                print(
                    f'        需要: '
                    f'{expected_periods}'
                )

                print(
                    f'        实际: '
                    f'{actual_periods}'
                )

                continue


        if len(temp) == 0:
            continue


        values = calculate_values(
            temp
        )


        row = {

            'k':
                k,

            'St':
                ST,

            'a':
                a,

            'dc_code':
                dc_code,

            'dc':
                DC_VALUE_DICT[
                    dc_code
                ],

            'sample_index':
                sample_index,

            'sample_type':
                'four_cycle_block',

            'period_start':
                p_start,

            'period_end':
                p_end,

            'sample_label':
                f'{p_start}-{p_end}',

            'n_base_cycles':
                4,
        }


        row.update(
            values
        )

        results.append(
            row
        )


    if len(results) == 0:
        return None


    return pd.DataFrame(
        results
    )


# ============================================================
# 处理单个工况
# ============================================================

def process_one_case(
    k,
    a,
    dc_code
):

    csv_path = get_csv_path(
        k=k,
        a=a,
        dc_code=dc_code
    )


    # --------------------------------------------------------
    # 文件检查
    # --------------------------------------------------------

    if not os.path.exists(
        csv_path
    ):

        print(
            f'[缺少文件] '
            f'k={k}, '
            f'a={a}, '
            f'DC={DC_VALUE_DICT[dc_code]}'
        )

        print(
            f'    {csv_path}'
        )

        return None


    print(
        f'\n正在处理: '
        f'k={k}, '
        f'St={ST}, '
        f'a={a}, '
        f'DC={DC_VALUE_DICT[dc_code]}'
    )


    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )


    # --------------------------------------------------------
    # 必需列
    # --------------------------------------------------------

    required_columns = [

        'period',

        # Lift
        'cl1_mean',
        'cl2_mean',
        'cl_total_mean',

        # Power
        'cp1_mean',
        'cp2_mean',
        'cp_total_mean',
    ]


    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]


    if len(
        missing_columns
    ) > 0:

        print(
            f'[跳过] 缺少列: '
            f'{missing_columns}'
        )

        print(
            f'文件：'
            f'{csv_path}'
        )

        return None


    # --------------------------------------------------------
    # 周期编号
    # --------------------------------------------------------

    df[
        'period'
    ] = pd.to_numeric(
        df[
            'period'
        ],
        errors='coerce'
    )


    df = df.dropna(
        subset=[
            'period'
        ]
    )


    df[
        'period'
    ] = (
        df[
            'period'
        ]
        .astype(int)
    )


    # --------------------------------------------------------
    # 根据 DC 选择统计方式
    # --------------------------------------------------------

    if dc_code == 1:

        return process_dc1(
            df=df,
            k=k,
            a=a,
            dc_code=dc_code
        )

    else:

        return process_intermittent_dc(
            df=df,
            k=k,
            a=a,
            dc_code=dc_code
        )


# ============================================================
# 批量处理
# ============================================================

all_sample_results = []


for k in K_LIST:

    a = CASE_DICT[
        k
    ]

    for dc_code in DC_CODES:

        temp_df = process_one_case(
            k=k,
            a=a,
            dc_code=dc_code
        )

        if temp_df is not None:

            all_sample_results.append(
                temp_df
            )


# ============================================================
# 检查
# ============================================================

if len(
    all_sample_results
) == 0:

    raise RuntimeError(
        '没有读取到任何有效数据。'
    )


# ============================================================
# 合并所有统计样本
# ============================================================

sample_df = pd.concat(
    all_sample_results,
    ignore_index=True
)


# ============================================================
# 保存所有独立统计样本
# ============================================================

sample_csv_path = os.path.join(
    save_root,
    'stacked_bar_statistical_samples.csv'
)


sample_df.to_csv(
    sample_csv_path,
    index=False,
    encoding='utf-8-sig',
    float_format='%.8e'
)


# ============================================================
# 输出实际采用周期
# ============================================================

print(
    '\n'
    '============================================================'
)

print(
    '各工况实际使用的统计周期'
)

print(
    '============================================================'
)


for (
    k,
    dc
), group in sample_df.groupby(
    [
        'k',
        'dc'
    ]
):

    labels = (
        group[
            'sample_label'
        ]
        .astype(str)
        .tolist()
    )

    print(
        f'k={k}, '
        f'DC={dc}: '
        f'{labels}'
    )


# ============================================================
# 最终统计
# ============================================================

METRICS = [

    # Lift
    'cl_fore',
    'cl_hind',
    'cl_system',

    # Power
    'cp_fore',
    'cp_hind',
    'cp_system',

    # Lift efficiency
    'eta_l_fore',
    'eta_l_hind',
    'eta_l_system',
]


summary_results = []


for (
    k,
    a,
    dc_code,
    dc
), group in sample_df.groupby(
    [
        'k',
        'a',
        'dc_code',
        'dc'
    ]
):

    result = {

        'k':
            k,

        'St':
            ST,

        'a':
            a,

        'dc_code':
            dc_code,

        'dc':
            dc,

        'n_samples':
            len(group),
    }


    if dc_code == 1:

        result[
            'statistical_method'
        ] = (
            'last_5_single_cycles'
        )

    else:

        result[
            'statistical_method'
        ] = (
            'five_4T_blocks'
        )


    # --------------------------------------------------------
    # 均值和样本标准差
    # --------------------------------------------------------

    for metric in METRICS:

        values = (
            group[
                metric
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
            .values
        )


        if len(values) == 0:

            mean_value = np.nan
            std_value = np.nan

        else:

            mean_value = np.mean(
                values
            )

            if len(values) > 1:

                std_value = np.std(
                    values,
                    ddof=1
                )

            else:

                std_value = np.nan


        result[
            f'mean_{metric}'
        ] = mean_value

        result[
            f'std_{metric}'
        ] = std_value


    summary_results.append(
        result
    )


summary_df = pd.DataFrame(
    summary_results
)


# ============================================================
# 排序
# ============================================================

order_dict = {
    dc_code: i
    for i, dc_code in enumerate(
        DC_PLOT_ORDER
    )
}


summary_df[
    'dc_order'
] = (
    summary_df[
        'dc_code'
    ]
    .map(
        order_dict
    )
)


summary_df = (
    summary_df
    .sort_values(
        by=[
            'k',
            'dc_order'
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 保存最终统计
# ============================================================

summary_csv_path = os.path.join(
    save_root,
    'stacked_bar_final_statistics.csv'
)


summary_df.to_csv(
    summary_csv_path,
    index=False,
    encoding='utf-8-sig',
    float_format='%.8e'
)


# ============================================================
# 获取指定 k 的绘图数据
# ============================================================

def get_k_data(k):

    temp = summary_df[
        np.isclose(
            summary_df[
                'k'
            ],
            k
        )
    ].copy()


    temp = temp.sort_values(
        'dc_order'
    )


    return temp


# ============================================================
# 通用坐标样式
# ============================================================

def style_axis(ax):

    x = np.arange(
        len(
            DC_PLOT_ORDER
        )
    )


    # --------------------------------------------------------
    # DC 刻度
    # --------------------------------------------------------

    ax.set_xticks(
        x
    )


    ax.set_xticklabels([
        '1',
        '0.75',
        '0.5',
        '0.25',
    ])


    # --------------------------------------------------------
    # 每个子图均显示 DC
    # --------------------------------------------------------

    ax.set_xlabel(
        r'$DC$'
    )


    # --------------------------------------------------------
    # 坐标刻度
    # --------------------------------------------------------

    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
        labelbottom=True
    )


    # --------------------------------------------------------
    # 四边框
    # --------------------------------------------------------

    ax.spines[
        'top'
    ].set_visible(
        True
    )

    ax.spines[
        'right'
    ].set_visible(
        True
    )

    ax.spines[
        'left'
    ].set_visible(
        True
    )

    ax.spines[
        'bottom'
    ].set_visible(
        True
    )


    # --------------------------------------------------------
    # 透明背景
    # --------------------------------------------------------

    ax.set_facecolor(
        'none'
    )

    ax.patch.set_facecolor(
        'none'
    )

    ax.patch.set_alpha(
        0.0
    )


# ============================================================
# 绘制堆叠柱 + 系统圆点折线
#
# 注意：
# 系统值现在直接使用正常系统性能值，
# 不再乘 2。
# ============================================================

# ============================================================
# 误差棒参数
# ============================================================

ERROR_CAPSIZE = 2.0
ERROR_LINEWIDTH = 0.7
ERROR_CAPTHICK = 0.7


# ============================================================
# 绘制堆叠柱 + 系统圆点折线 + 误差棒
# ============================================================

def plot_stacked(
    ax,
    temp,
    fore_col,
    hind_col,
    system_col,
    fore_std_col,
    hind_std_col,
    system_std_col,
    ylabel
):

    x = np.arange(
        len(temp)
    )


    # ========================================================
    # 均值
    # ========================================================

    fore = temp[
        fore_col
    ].to_numpy(
        dtype=float
    )

    hind = temp[
        hind_col
    ].to_numpy(
        dtype=float
    )

    system = temp[
        system_col
    ].to_numpy(
        dtype=float
    )


    # ========================================================
    # 标准差
    # ========================================================

    fore_err = temp[
        fore_std_col
    ].to_numpy(
        dtype=float
    )

    hind_err = temp[
        hind_std_col
    ].to_numpy(
        dtype=float
    )

    system_err = temp[
        system_std_col
    ].to_numpy(
        dtype=float
    )


    # NaN 标准差转换为 0，避免 matplotlib 报错
    fore_err = np.nan_to_num(
        fore_err,
        nan=0.0
    )

    hind_err = np.nan_to_num(
        hind_err,
        nan=0.0
    )

    system_err = np.nan_to_num(
        system_err,
        nan=0.0
    )


    # ========================================================
    # 前翼柱
    # ========================================================

    ax.bar(
        x,
        fore,

        width=BAR_WIDTH,

        color=FORE_COLOR,

        edgecolor=
            BAR_EDGE_COLOR,

        linewidth=
            BAR_LINEWIDTH,

        zorder=2
    )


    # ========================================================
    # 后翼柱
    # ========================================================

    ax.bar(
        x,
        hind,

        width=BAR_WIDTH,

        bottom=fore,

        color=HIND_COLOR,

        edgecolor=
            BAR_EDGE_COLOR,

        linewidth=
            BAR_LINEWIDTH,

        zorder=2
    )


    # ========================================================
    # 前翼误差棒
    #
    # 中心位置：
    #     C_F
    # ========================================================

    ax.errorbar(
        x,
        fore,

        yerr=fore_err,

        fmt='none',

        ecolor='black',

        elinewidth=
            ERROR_LINEWIDTH,

        capsize=
            ERROR_CAPSIZE,

        capthick=
            ERROR_CAPTHICK,

        zorder=5
    )


    # ========================================================
    # 后翼误差棒
    #
    # 因为后翼采用堆叠柱，因此其柱顶位置为：
    #
    #     fore + hind
    #
    # 误差大小仍然使用 hind 自身标准差。
    # ========================================================

    stacked_top = (
        fore +
        hind
    )

    ax.errorbar(
        x,
        stacked_top,

        yerr=hind_err,

        fmt='none',

        ecolor='black',

        elinewidth=
            ERROR_LINEWIDTH,

        capsize=
            ERROR_CAPSIZE,

        capthick=
            ERROR_CAPTHICK,

        zorder=5
    )


    # ========================================================
    # 系统性能 + 系统误差棒
    #
    # 直接使用正常系统值：
    #
    # Lift:
    #     C_LS
    #
    # Power:
    #     C_PS
    #
    # Efficiency:
    #     eta_LS
    # ========================================================

    ax.errorbar(
        x,
        system,

        yerr=system_err,

        color=
            SYSTEM_LINE_COLOR,

        ecolor=
            SYSTEM_LINE_COLOR,

        linewidth=
            SYSTEM_LINEWIDTH,

        elinewidth=
            ERROR_LINEWIDTH,

        capsize=
            ERROR_CAPSIZE,

        capthick=
            ERROR_CAPTHICK,

        marker='o',

        markersize=
            SYSTEM_MARKERSIZE,

        markerfacecolor=
            'white',

        markeredgecolor=
            'black',

        markeredgewidth=
            0.9,

        linestyle='-',

        zorder=6
    )


    # ========================================================
    # y 轴标签
    # ========================================================

    ax.set_ylabel(
        ylabel,
        rotation=0
    )

    ax.yaxis.set_label_coords(
        -0.17,
        0.5
    )


    # ========================================================
    # y = 0
    # ========================================================

    y_min, y_max = ax.get_ylim()

    if y_min < 0 < y_max:

        ax.axhline(
            0,

            color='black',

            linewidth=0.5,

            linestyle='--',

            zorder=1
        )


# ============================================================
# 最终绘图
#
# 4 行 × 3 列
#
# (a) k = 0.8
# (b) k = 1.1
# (c) k = 1.4
# (d) k = 1.7
#
# 第一列：Lift
# 第二列：Power
# 第三列：Lift efficiency
# ============================================================

fig, axes = plt.subplots(
    nrows=4,
    ncols=3,

    figsize=(
        6.5,
        6.5
    ),

    sharex='col',
    sharey='col'
)


# ============================================================
# 整张图透明背景
# ============================================================

fig.patch.set_facecolor(
    'none'
)

fig.patch.set_alpha(
    0.0
)


# ============================================================
# 子图编号
# ============================================================

SUBPLOT_LABELS = [
    'a',
    'b',
    'c',
    'd',
]


# ============================================================
# 逐行绘图
# ============================================================

for row_idx, k in enumerate(
    K_LIST
):

    temp = get_k_data(
        k
    )


    if len(temp) == 0:

        print(
            f'[警告] '
            f'k={k} '
            f'没有可用于绘图的数据'
        )

        continue


    # ========================================================
    # 第一列：Lift
    #
    # 柱：
    #   C_LF + C_LH
    #
    # 圆点：
    #   C_LS
    # ========================================================

    plot_stacked(

        ax=
        axes[
            row_idx,
            0
        ],

        temp=temp,

        fore_col=
        'mean_cl_fore',

        hind_col=
        'mean_cl_hind',

        system_col=
        'mean_cl_system',

        fore_std_col=
        'std_cl_fore',

        hind_std_col=
        'std_cl_hind',

        system_std_col=
        'std_cl_system',

        ylabel=
        r'$\overline{C}_{L}$',
    )

    # ========================================================
    # 第二列：Power
    #
    # 柱：
    #   C_PF + C_PH
    #
    # 圆点：
    #   C_PS
    # ========================================================

    plot_stacked(

        ax=
        axes[
            row_idx,
            1
        ],

        temp=temp,

        fore_col=
        'mean_cp_fore',

        hind_col=
        'mean_cp_hind',

        system_col=
        'mean_cp_system',

        fore_std_col=
        'std_cp_fore',

        hind_std_col=
        'std_cp_hind',

        system_std_col=
        'std_cp_system',

        ylabel=
        r'$\overline{C}_{P}$',
    )


    # ========================================================
    # 第三列：Lift efficiency
    #
    # 前翼：
    #   eta_LF = C_LF / C_PF
    #
    # 后翼：
    #   eta_LH = C_LH / C_PH
    #
    # 系统：
    #   eta_LS =
    #   (C_LF + C_LH)/(C_PF + C_PH)
    # ========================================================

    plot_stacked(

        ax=
        axes[
            row_idx,
            2
        ],

        temp=temp,

        fore_col=
        'mean_eta_l_fore',

        hind_col=
        'mean_eta_l_hind',

        system_col=
        'mean_eta_l_system',

        fore_std_col=
        'std_eta_l_fore',

        hind_std_col=
        'std_eta_l_hind',

        system_std_col=
        'std_eta_l_system',

        ylabel=
        r'$\overline{\eta}_{L}$',
    )


    # ========================================================
    # 每个子图均显示 DC
    # ========================================================

    for col_idx in range(
        3
    ):

        style_axis(
            axes[
                row_idx,
                col_idx
            ]
        )


    # ========================================================
    # 每行左侧标注 (a)-(d)
    # ========================================================

    axes[
        row_idx,
        0
    ].text(

        -0.31,
        0.99,

        rf'$\it{{({SUBPLOT_LABELS[row_idx]})}}$',

        transform=
            axes[
                row_idx,
                0
            ].transAxes,

        ha='left',
        va='bottom'
    )


# ============================================================
# sharex 会隐藏部分横坐标
# 强制全部显示
# ============================================================

for row_idx in range(
    4
):

    for col_idx in range(
        3
    ):

        axes[
            row_idx,
            col_idx
        ].tick_params(
            axis='x',
            labelbottom=True
        )


# ============================================================
# 图例
#
# 放在左上角 (a) 子图
# ============================================================

legend_handles = [

    Patch(
        facecolor=
            FORE_COLOR,

        edgecolor=
            'black',

        linewidth=
            BAR_LINEWIDTH,

        label=
            r'Forewing'
    ),


    Patch(
        facecolor=
            HIND_COLOR,

        edgecolor=
            'black',

        linewidth=
            BAR_LINEWIDTH,

        label=
            r'Hindwing'
    ),


    Line2D(
        [0],
        [0],

        color=
            'black',

        linewidth=
            SYSTEM_LINEWIDTH,

        marker=
            'o',

        markersize=
            SYSTEM_MARKERSIZE,

        markerfacecolor=
            'white',

        markeredgecolor=
            'black',

        markeredgewidth=
            0.9,

        label=
            r'System'
    ),
]


axes[
    0,
    0
].legend(

    handles=
        legend_handles,

    loc=
        'upper left',

    frameon=
        False,

    ncol=
        1,

    handlelength=
        1.6,

    handletextpad=
        0.5,

    borderaxespad=
        0.4,

    labelspacing=
        0.35,

    columnspacing=
        0.8
)


# ============================================================
# 布局
# ============================================================

plt.tight_layout(

    rect=[
        0.02,
        0.01,
        1.00,
        1.00
    ],

    pad=
        0.7,

    w_pad=
        1.0,

    h_pad=
        0.2
)


# ============================================================
# 保存路径
# ============================================================

png_path = os.path.join(
    save_root,
    'stacked_lift_power_efficiency_vs_dc.png'
)

svg_path = os.path.join(
    save_root,
    'stacked_lift_power_efficiency_vs_dc.svg'
)

pdf_path = os.path.join(
    save_root,
    'stacked_lift_power_efficiency_vs_dc.pdf'
)


# ============================================================
# PNG
# ============================================================

fig.savefig(
    png_path,

    dpi=600,

    bbox_inches='tight',

    transparent=True,

    facecolor='none',

    edgecolor='none'
)


# ============================================================
# SVG
# ============================================================

fig.savefig(
    svg_path,

    format='svg',

    bbox_inches='tight',

    transparent=True,

    facecolor='none',

    edgecolor='none'
)


# ============================================================
# PDF
# ============================================================

# fig.savefig(
#     pdf_path,
#     format='pdf',
#     bbox_inches='tight',
#     transparent=True,
#     facecolor='none',
#     edgecolor='none'
# )


# ============================================================
# 显示
# ============================================================

plt.show()


# ============================================================
# 完成
# ============================================================

print(
    '\n'
    '============================================================'
)

print(
    '全部处理完成'
)

print(
    '============================================================'
)


print(
    f'\nDC=1：'
    f'最后 {DC1_LAST_N_PERIODS} 个普通周期'
)


print(
    f'DC<1：'
    f'{BIG_CYCLES}'
)


print(
    f'\n统计样本：\n'
    f'{sample_csv_path}'
)


print(
    f'\n最终统计：\n'
    f'{summary_csv_path}'
)


print(
    f'\nPNG：\n'
    f'{png_path}'
)


print(
    f'\nSVG：\n'
    f'{svg_path}'
)