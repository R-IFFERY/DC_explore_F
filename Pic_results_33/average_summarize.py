# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    # --------------------------------------------------------
    # 透明背景
    # --------------------------------------------------------
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
    'large_cycle_system_results'
)

os.makedirs(save_root, exist_ok=True)


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
# 间歇拍动的大周期设置
#
# 仅用于 DC != 1
# ============================================================

BIG_CYCLES = [
    # (9, 12),
    (13, 16),
    (17, 20),
    (21, 24),
    (25, 28),
    (29, 32)
]

REQUIRE_COMPLETE_BIG_CYCLE = True


# ============================================================
# DC = 1 的统计设置
# ============================================================

DC1_LAST_N_PERIODS = 5


# ============================================================
# 绘图样式
#
# 所有曲线黑色
# 不同 k 通过 marker 和 linestyle 区分
# ============================================================

K_STYLE = {

    0.8: {
        'marker': 'o',
        'linestyle': '-',
    },

    1.1: {
        'marker': 's',
        'linestyle': '--',
    },

    1.4: {
        'marker': '^',
        'linestyle': '-.',
    },

    1.7: {
        'marker': 'D',
        'linestyle': ':',
    },
}

LINEWIDTH = 0.8
MARKERSIZE = 3.5

CAPSIZE = 2.0
ERROR_LINEWIDTH = 0.8


# ============================================================
# 数字格式
# ============================================================

def format_num(x):
    """
    文件名中的数字格式。

    例如：
        0.8 -> '0.8'
        1.0 -> '1'
    """

    if float(x).is_integer():
        return str(int(x))

    return f'{x:g}'


# ============================================================
# 生成 CSV 路径
# ============================================================

def get_csv_path(k, a, dc_code):

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
# 计算单个统计单元的系统参数
# ============================================================

def calculate_system_values(temp):
    """
    temp 可以是：

    1. DC=1 时的单个普通周期；
    2. DC<1 时的一个 4T 大周期。

    返回：
        CL
        CT
        CP
        eta_L
        eta_T
    """

    cl = temp[
        'cl_total_mean'
    ].mean()

    ct = temp[
        'ct_total_mean'
    ].mean()

    cp = temp[
        'cp_total_mean'
    ].mean()

    # --------------------------------------------------------
    # 效率
    # --------------------------------------------------------

    if np.isclose(cp, 0.0):

        eta_l = np.nan
        eta_t = np.nan

    else:

        eta_l = cl / cp
        eta_t = ct / cp

    return (
        cl,
        ct,
        cp,
        eta_l,
        eta_t
    )


# ============================================================
# DC = 1
#
# 使用最后 5 个普通周期
# ============================================================

def process_dc1(df, k, a, dc_code):

    # --------------------------------------------------------
    # 获取实际存在的所有周期
    # --------------------------------------------------------

    available_periods = sorted(
        df['period']
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if len(available_periods) < DC1_LAST_N_PERIODS:

        print(
            f'[警告] k={k}, DC=1 '
            f'只有 {len(available_periods)} 个有效周期，'
            f'不足 {DC1_LAST_N_PERIODS} 个'
        )

        return None

    # --------------------------------------------------------
    # 最后五个周期
    # --------------------------------------------------------

    selected_periods = available_periods[
        -DC1_LAST_N_PERIODS:
    ]

    print(
        f'    DC=1 使用最后 {DC1_LAST_N_PERIODS} 个周期: '
        f'{selected_periods}'
    )

    results = []

    # --------------------------------------------------------
    # 每个普通周期作为一个独立统计样本
    # --------------------------------------------------------

    for sample_index, period_idx in enumerate(
        selected_periods,
        start=1
    ):

        temp = df[
            df['period'] == period_idx
        ].copy()

        if len(temp) == 0:
            continue

        (
            cl,
            ct,
            cp,
            eta_l,
            eta_t
        ) = calculate_system_values(
            temp
        )

        results.append({

            'k': k,
            'St': ST,
            'a': a,

            'dc_code': dc_code,
            'dc': DC_VALUE_DICT[dc_code],

            'sample_index': sample_index,

            'sample_type': 'single_cycle',

            'period_start': period_idx,
            'period_end': period_idx,

            'sample_label':
                f'{period_idx}',

            'n_base_cycles': 1,

            'cl': cl,
            'ct': ct,
            'cp': cp,

            'eta_l': eta_l,
            'eta_t': eta_t,
        })

    if len(results) == 0:
        return None

    return pd.DataFrame(
        results
    )


# ============================================================
# DC < 1
#
# 使用五个 4T 大周期
# ============================================================

def process_intermittent_dc(
    df,
    k,
    a,
    dc_code
):

    results = []

    for sample_index, (
        p_start,
        p_end
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
            df['period'].isin(
                expected_periods
            )
        ].copy()

        actual_periods = sorted(
            temp['period']
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        # ----------------------------------------------------
        # 一个大周期必须有完整 4 个普通周期
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

        (
            cl,
            ct,
            cp,
            eta_l,
            eta_t
        ) = calculate_system_values(
            temp
        )

        results.append({

            'k': k,
            'St': ST,
            'a': a,

            'dc_code': dc_code,
            'dc': DC_VALUE_DICT[dc_code],

            'sample_index': sample_index,

            'sample_type': 'four_cycle_block',

            'period_start': p_start,
            'period_end': p_end,

            'sample_label':
                f'{p_start}-{p_end}',

            'n_base_cycles': 4,

            'cl': cl,
            'ct': ct,
            'cp': cp,

            'eta_l': eta_l,
            'eta_t': eta_t,
        })

    if len(results) == 0:
        return None

    return pd.DataFrame(
        results
    )


# ============================================================
# 处理一个工况
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

    if not os.path.exists(csv_path):

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
    # 读取
    # --------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # 必要列检查
    # --------------------------------------------------------

    required_columns = [
        'period',
        'cl_total_mean',
        'ct_total_mean',
        'cp_total_mean',
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if len(missing_columns) > 0:

        print(
            f'[跳过] 缺少列: '
            f'{missing_columns}'
        )

        return None

    # --------------------------------------------------------
    # 周期列处理
    # --------------------------------------------------------

    df['period'] = pd.to_numeric(
        df['period'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['period']
    )

    df['period'] = (
        df['period']
        .astype(int)
    )

    # ========================================================
    # 根据 DC 采用不同统计方式
    # ========================================================

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
# 批量处理所有工况
# ============================================================

all_sample_results = []

for k in K_LIST:

    a = CASE_DICT[k]

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

if len(all_sample_results) == 0:

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
# 保存所有统计单元
# ============================================================

sample_csv_path = os.path.join(
    save_root,
    'system_statistical_samples.csv'
)

sample_df.to_csv(
    sample_csv_path,
    index=False,
    encoding='utf-8-sig',
    float_format='%.8e'
)


# ============================================================
# 打印具体采用的数据
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
        f'k={k}, DC={dc}: '
        f'{labels}'
    )


# ============================================================
# 最终统计
# ============================================================

METRICS = [
    'cl',
    'ct',
    'cp',
    'eta_l',
    'eta_t',
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

        'k': k,
        'St': ST,
        'a': a,

        'dc_code': dc_code,
        'dc': dc,

        'n_samples':
            len(group),
    }

    # --------------------------------------------------------
    # 统计方法标记
    # --------------------------------------------------------

    if dc_code == 1:

        result[
            'statistical_method'
        ] = 'last_5_single_cycles'

    else:

        result[
            'statistical_method'
        ] = 'five_4T_blocks'

    # --------------------------------------------------------
    # 均值和样本标准差
    # --------------------------------------------------------

    for metric in METRICS:

        values = (
            group[
                metric
            ]
            .replace(
                [np.inf, -np.inf],
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

summary_df = summary_df.sort_values(
    by=[
        'k',
        'dc'
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# 保存最终统计
# ============================================================

summary_csv_path = os.path.join(
    save_root,
    'system_final_statistics.csv'
)

summary_df.to_csv(
    summary_csv_path,
    index=False,
    encoding='utf-8-sig',
    float_format='%.8e'
)


# ============================================================
# 获取绘图数据
# ============================================================

def get_plot_data(
    k,
    metric
):

    temp = summary_df[
        summary_df['k'] == k
    ].copy()

    order_dict = {
        dc_code: i
        for i, dc_code in enumerate(
            DC_PLOT_ORDER
        )
    }

    temp[
        'dc_order'
    ] = temp[
        'dc_code'
    ].map(
        order_dict
    )

    temp = temp.sort_values(
        'dc_order'
    )

    x = temp[
        'dc'
    ].values

    y = temp[
        f'mean_{metric}'
    ].values

    yerr = temp[
        f'std_{metric}'
    ].values

    return (
        x,
        y,
        yerr
    )


# ============================================================
# 单个子图绘制
# ============================================================

def plot_metric(
    ax,
    metric,
    ylabel,
    subplot_label,
    ylim=None,
    show_zero_line=False,
    ylabel_x=-0.13
):

    # --------------------------------------------------------
    # 每个 k 一条线
    # --------------------------------------------------------

    for k in K_LIST:

        x, y, yerr = get_plot_data(
            k=k,
            metric=metric
        )

        if len(x) == 0:
            continue

        style = K_STYLE[k]

        ax.errorbar(
            x,
            y,

            yerr=yerr,

            marker=style['marker'],
            linestyle=style['linestyle'],

            # ------------------------------------------------
            # 统一黑色
            # ------------------------------------------------
            color='black',
            ecolor='black',

            # ------------------------------------------------
            # 空心数据点
            # ------------------------------------------------
            markerfacecolor='none',
            markeredgecolor='black',
            markeredgewidth=0.9,

            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,

            capsize=CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,

            label=rf'$k={k}$',

            zorder=3
        )

    # ========================================================
    # 横坐标
    # ========================================================

    ax.set_xlabel(
        r'$DC$'
    )

    ax.set_xticks(
        [
            1.00,
            0.75,
            0.50,
            0.25
        ]
    )

    ax.set_xticklabels(
        [
            '1',
            '0.75',
            '0.5',
            '0.25'
        ]
    )

    # DC=1 位于左侧
    ax.set_xlim(
        1.05,
        0.20
    )

    # ========================================================
    # 纵坐标
    # ========================================================

    ax.set_ylabel(
        ylabel,
        rotation=0
    )

    # --------------------------------------------------------
    # 每张子图单独控制纵坐标标签位置
    #
    # 数值越负，标签越向左
    # --------------------------------------------------------

    ax.yaxis.set_label_coords(
        ylabel_x,
        0.5
    )

    # 设置 y 轴范围
    if ylim is not None:

        ax.set_ylim(
            ylim
        )

    # ========================================================
    # y=0 参考线
    # ========================================================

    if show_zero_line:

        ax.axhline(
            y=0,
            color='black',
            linewidth=0.7,
            linestyle='-.',
            dashes=(8, 4, 2, 4),
            zorder=1
        )

    # ========================================================
    # 坐标样式
    # ========================================================

    ax.tick_params(
        axis='both',
        direction='in'
    )

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # --------------------------------------------------------
    # 子图背景完全透明
    # --------------------------------------------------------

    ax.set_facecolor('none')
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0.0)

    # ========================================================
    # 子图编号
    # ========================================================

    ax.text(
        -0.25,
        1.04,
        rf'$\it{{({subplot_label})}}$',

        transform=ax.transAxes,

        ha='left',
        va='bottom'
    )


# ============================================================
# 最终绘图
#
# 2 × 3
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(6.5, 4.2)
)

# ============================================================
# 整张图片背景完全透明
# ============================================================

fig.patch.set_facecolor('none')
fig.patch.set_alpha(0.0)


# ============================================================
# (a) Lift
# ============================================================

plot_metric(
    axes[0, 0],
    metric='cl',
    ylabel=r'$\overline{C}_{LS}$',
    subplot_label='a',
    ylim=(0.5, 2.5),
    show_zero_line=False,

    # 向左移动，避免与刻度重叠
    ylabel_x=-0.19
)


# ============================================================
# (b) Thrust
# ============================================================

plot_metric(
    axes[0, 1],
    metric='ct',
    ylabel=r'$\overline{C}_{TS}$',
    subplot_label='b',
    ylim=(-0.15, 0.25),
    show_zero_line=True,

    # 向左移动，避免与刻度重叠
    ylabel_x=-0.19
)


# ============================================================
# (c) Power
# ============================================================

plot_metric(
    axes[0, 2],
    metric='cp',
    ylabel=r'$\overline{C}_{PS}$',
    subplot_label='c',
    ylim=(0.0, 2.5),
    show_zero_line=False,

    # 保持原位置
    ylabel_x=-0.13
)


# ============================================================
# (d) Lift efficiency
# ============================================================

plot_metric(
    axes[1, 0],
    metric='eta_l',
    ylabel=r'$\eta_L$',
    subplot_label='d',
    ylim=(0.0, 2.0),
    show_zero_line=False,

    # 稍向左移动
    ylabel_x=-0.17
)


# ============================================================
# (e) Thrust efficiency
# ============================================================

plot_metric(
    axes[1, 1],
    metric='eta_t',
    ylabel=r'$\eta_T$',
    subplot_label='e',
    ylim=(-0.2, 0.2),
    show_zero_line=True,

    # 稍向左移动
    ylabel_x=-0.17
)


# ============================================================
# 空白子图
# ============================================================

axes[1, 2].axis(
    'off'
)

axes[1, 2].set_facecolor(
    'none'
)

axes[1, 2].patch.set_facecolor(
    'none'
)

axes[1, 2].patch.set_alpha(
    0.0
)


# ============================================================
# 图例
# ============================================================

handles, labels = (
    axes[0, 0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,

    loc='lower right',

    bbox_to_anchor=(
        0.96,
        0.13
    ),

    frameon=False,

    ncol=1,

    handlelength=2.3,

    labelspacing=0.4
)


# ============================================================
# 布局
# ============================================================

plt.tight_layout(
    pad=0.8,
    w_pad=1.0,
    h_pad=1.1
)


# ============================================================
# 保存
# ============================================================

png_path = os.path.join(
    save_root,
    'system_vs_dc.png'
)

svg_path = os.path.join(
    save_root,
    'system_vs_dc.svg'
)

pdf_path = os.path.join(
    save_root,
    'system_vs_dc.pdf'
)


# ============================================================
# PNG：完全透明背景
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
# SVG：完全透明背景
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
# PDF：完全透明背景
# ============================================================

fig.savefig(
    pdf_path,
    format='pdf',
    bbox_inches='tight',
    transparent=True,
    facecolor='none',
    edgecolor='none'
)


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
    '\n统计方法：'
)

print(
    f'DC=1    : 使用数据最后 {DC1_LAST_N_PERIODS} 个普通周期'
)

print(
    f'DC=0.75 : 使用 {BIG_CYCLES}'
)

print(
    f'DC=0.50 : 使用 {BIG_CYCLES}'
)

print(
    f'DC=0.25 : 使用 {BIG_CYCLES}'
)

print(
    f'\n统计样本数据：\n'
    f'{sample_csv_path}'
)

print(
    f'\n最终统计数据：\n'
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

print(
    f'\nPDF：\n'
    f'{pdf_path}'
)